#include "storage/ducklake_format_reader.hpp"
#include "storage/ducklake_multi_file_reader.hpp"
#include "storage/ducklake_scan.hpp"
#include "storage/ducklake_delete_filter.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

DuckLakeFormatReader::DuckLakeFormatReader(DuckLakeFunctionInfo &read_info, const OpenFileInfo &info, string format_p,
                                           vector<MultiFileColumnDefinition> columns_p)
    : BaseFileReader(info), read_info(read_info), format(std::move(format_p)) {
	columns = std::move(columns_p);
}

TableFunction DuckLakeFormatReader::GetReadFunction(ClientContext &context, const string &format) {
	auto &instance = DatabaseInstance::GetDatabase(context);
	ExtensionHelper::TryAutoLoadExtension(instance, format);
	ExtensionLoader loader(instance, "ducklake");
	auto &scan_entry = loader.GetTableFunction(Identifier("read_" + format));
	return *scan_entry.functions.functions[0];
}

bool DuckLakeFormatReader::TryInitializeScan(ClientContext &context, GlobalTableFunctionState &gstate,
                                             LocalTableFunctionState &lstate) {
	{
		lock_guard<mutex> guard(lock);
		if (initialized_scan) {
			return false;
		}
		initialized_scan = true;
	}

	// bind the file to discover its schema
	file_scan = GetReadFunction(context, format);

	vector<Value> children;
	children.push_back(Value(file.path));
	named_argument_map_t named_params;
	vector<LogicalType> input_types;
	vector<Identifier> input_names;
	if (StringUtil::CIEquals(format, "json")) {
		// read_json only auto-detects when asked; the SQL wrapper does this for us
		named_params[Identifier("auto_detect")] = Value::BOOLEAN(true);
	}

	TableFunctionRef empty;
	BoundTableFunction bound_table_function(file_scan);
	TableFunctionBindInput bind_input(children, named_params, input_types, input_names, file_scan.function_info.get(),
	                                  nullptr, bound_table_function, empty);
	bind_data = file_scan.bind(context, bind_input, file_types, file_names);

	// map each projected column onto a file column (by name), the ordinal, or NULL if absent
	vector<LogicalType> scan_types;
	for (auto &column_id : column_indexes) {
		auto &col = columns[column_id.GetPrimaryIndex()];
		if (!col.identifier.IsNull() && col.identifier.type().id() == LogicalTypeId::INTEGER &&
		    IntegerValue::Get(col.identifier) == MultiFileReader::ORDINAL_FIELD_ID) {
			// file_row_number; row_id/snapshot_id are derived from it by the MultiFileReader
			output_kinds.push_back(FormatReaderColumn::ORDINAL);
			output_source_index.push_back(0);
			continue;
		}
		optional_idx file_position;
		for (idx_t i = 0; i < file_names.size(); i++) {
			if (StringUtil::CIEquals(file_names[i].GetIdentifierName(), col.name.GetIdentifierName())) {
				file_position = i;
				break;
			}
		}
		if (!file_position.IsValid()) {
			output_kinds.push_back(FormatReaderColumn::MISSING);
			output_source_index.push_back(0);
			continue;
		}
		output_kinds.push_back(FormatReaderColumn::FILE_COLUMN);
		output_source_index.push_back(scan_column_ids.size());
		scan_column_ids.push_back(file_position.GetIndex());
		scan_types.push_back(file_types[file_position.GetIndex()]);
	}
	// need at least one projected column so the scan can report a chunk cardinality
	if (scan_column_ids.empty() && !file_types.empty()) {
		scan_column_ids.push_back(0);
		scan_types.push_back(file_types[0]);
	}
	scan_chunk.Initialize(context, scan_types);

	for (auto &entry : expression_map) {
		auto &column_index = column_indexes[entry.first];
		auto column_id = column_index.GetPrimaryIndex();
		auto &expression_data = entry.second;
		auto &expression = *expression_data.expression;
		expression_executors[column_id] = make_uniq<ExpressionExecutor>(context, expression);
	}

	thread_context = make_uniq<ThreadContext>(context);
	execution_context = make_uniq<ExecutionContext>(context, *thread_context, nullptr);
	TableFunctionInitInput init_input(bind_data.get(), scan_column_ids, vector<idx_t>(), nullptr);
	global_state = file_scan.init_global(context, init_input);
	local_state = file_scan.init_local(*execution_context, init_input, global_state.get());
	return true;
}

bool DuckLakeFormatReader::TryEvaluateExpression(ClientContext &context, idx_t virtual_col_idx, Vector &input_vector,
                                                 const LogicalType &input_type, idx_t count, Vector &output_vector) {
	if (expression_map.empty() || virtual_col_idx >= column_ids.size()) {
		return false;
	}
	auto local_id = column_ids[MultiFileLocalIndex(virtual_col_idx)];
	auto expr_it = expression_executors.find(local_id);
	if (expr_it == expression_executors.end()) {
		return false;
	}
	DataChunk expr_input;
	expr_input.Initialize(Allocator::Get(context), {input_type});
	expr_input.Reset();
	expr_input.data[0].Reference(input_vector);
	expr_input.SetChildCardinality(count);
	expr_it->second->ExecuteExpression(expr_input, output_vector);
	return true;
}

AsyncResult DuckLakeFormatReader::Scan(ClientContext &context, GlobalTableFunctionState &global_table_state,
                                       LocalTableFunctionState &local_table_state, DataChunk &chunk) {
	scan_chunk.Reset();
	TableFunctionInput function_input(bind_data.get(), local_state.get(), global_state.get());
	file_scan.function(context, function_input, scan_chunk);
	idx_t scan_count = scan_chunk.size();
	if (scan_count == 0) {
		return AsyncResult(SourceResultType::FINISHED);
	}

	for (idx_t c = 0; c < output_kinds.size(); c++) {
		switch (output_kinds[c]) {
		case FormatReaderColumn::FILE_COLUMN: {
			auto &source = scan_chunk.data[output_source_index[c]];
			if (TryEvaluateExpression(context, c, source, source.GetType(), scan_count, chunk.data[c])) {
				break;
			}
			if (chunk.data[c].GetType() != source.GetType()) {
				VectorOperations::Cast(context, source, chunk.data[c], scan_count);
			} else {
				chunk.data[c].Reference(source);
			}
			break;
		}
		case FormatReaderColumn::ORDINAL: {
			Vector ordinal_vector(LogicalType::BIGINT);
			VectorOperations::GenerateSequence(ordinal_vector, scan_count, file_row_number);
			if (TryEvaluateExpression(context, c, ordinal_vector, LogicalType::BIGINT, scan_count, chunk.data[c])) {
				break;
			}
			VectorOperations::Copy(ordinal_vector, chunk.data[c], scan_count, 0, 0);
			break;
		}
		case FormatReaderColumn::MISSING:
			chunk.data[c].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(chunk.data[c], true);
			break;
		}
	}
	chunk.SetChildCardinality(scan_count);

	if (filters || deletion_filter) {
		SelectionVector sel;
		idx_t approved_tuple_count = scan_count;
		if (deletion_filter) {
			approved_tuple_count = deletion_filter->Filter(file_row_number, approved_tuple_count, sel);
		}
		if (filters) {
			for (auto &entry : *filters) {
				auto &filter = entry.Filter();
				if (ExpressionFilter::IsRootOptionalFilter(filter)) {
					continue;
				}
				auto column_id = entry.GetIndex().GetIndex();
				auto filter_state = TableFilterState::Initialize(context, filter);
				approved_tuple_count = ColumnSegment::FilterSelection(sel, chunk.data[column_id], *filter_state,
				                                                      chunk.size(), approved_tuple_count);
			}
		}
		if (approved_tuple_count != chunk.size()) {
			chunk.Slice(sel, approved_tuple_count);
		}
	}

	file_row_number += NumericCast<int64_t>(scan_count);
	return AsyncResult(SourceResultType::HAVE_MORE_OUTPUT);
}

void DuckLakeFormatReader::AddVirtualColumn(column_t virtual_column_id) {
	if (virtual_column_id == MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER) {
		columns.back().identifier = Value::INTEGER(MultiFileReader::ORDINAL_FIELD_ID);
	} else {
		throw InternalException("Unsupported virtual column id %d for format reader", virtual_column_id);
	}
}

string DuckLakeFormatReader::GetReaderType() const {
	return "DuckLake " + format;
}

} // namespace duckdb
