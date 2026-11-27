#include "storage/ducklake_format_reader.hpp"
#include "storage/ducklake_multi_file_reader.hpp"
#include "storage/ducklake_scan.hpp"
#include "storage/ducklake_delete_filter.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/storage/table/column_segment.hpp"

namespace duckdb {

static void ExtractStructField(ClientContext &context, Vector &source, const vector<idx_t> &path, idx_t count,
                               Vector &result) {
	ValidityMask parent_validity;
	bool has_parent_nulls = false;
	Vector *cur = &source;
	for (auto child_idx : path) {
		if (cur->GetVectorType() == VectorType::CONSTANT_VECTOR) {
			if (ConstantVector::IsNull(*cur)) {
				result.SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result, true);
				return;
			}
			cur = &StructVector::GetEntries(*cur)[child_idx];
			continue;
		}
		if (cur->GetVectorType() != VectorType::FLAT_VECTOR) {
			cur->Flatten();
		}
		auto &validity = FlatVector::Validity(*cur);
		if (!validity.CannotHaveNull()) {
			if (!has_parent_nulls) {
				parent_validity.Initialize(count);
				has_parent_nulls = true;
			}
			parent_validity.Combine(validity, count);
		}
		cur = &StructVector::GetEntries(*cur)[child_idx];
	}
	if (result.GetType() != cur->GetType()) {
		VectorOperations::Cast(context, *cur, result, count);
	} else if (!has_parent_nulls) {
		result.Reference(*cur);
		return;
	} else {
		VectorOperations::Copy(*cur, result, count, 0, 0);
	}
	if (has_parent_nulls) {
		result.Flatten();
		FlatVector::ValidityMutable(result).Combine(parent_validity, count);
	}
}

static idx_t FilterByRowNumbers(const DuckLakeDeleteFilter &delete_filter, const Vector &row_numbers, idx_t count,
                                SelectionVector &result_sel) {
	auto &deleted_rows = delete_filter.delete_data->deleted_rows;
	if (deleted_rows.empty()) {
		return count;
	}
	auto &snapshot_ids = delete_filter.delete_data->snapshot_ids;
	bool check_snapshots = delete_filter.snapshot_filter.IsValid() && !snapshot_ids.empty();
	bool is_unsigned = row_numbers.GetType().id() == LogicalTypeId::UBIGINT;

	idx_t delete_idx = 0;
	idx_t prev_row = 0;
	bool initialized_sel = false;
	idx_t result_count = 0;

	auto check_row = [&](idx_t i, idx_t pos) {
		if (i == 0 || pos < prev_row) {
			auto it = std::lower_bound(deleted_rows.begin(), deleted_rows.end(), pos);
			delete_idx = NumericCast<idx_t>(it - deleted_rows.begin());
		} else if (delete_idx < deleted_rows.size() && deleted_rows[delete_idx] < pos) {
			auto it =
			    std::lower_bound(deleted_rows.begin() + NumericCast<int64_t>(delete_idx), deleted_rows.end(), pos);
			delete_idx = NumericCast<idx_t>(it - deleted_rows.begin());
		}
		prev_row = pos;

		bool is_deleted = false;
		if (delete_idx < deleted_rows.size() && deleted_rows[delete_idx] == pos) {
			is_deleted = !check_snapshots || snapshot_ids[delete_idx] <= delete_filter.snapshot_filter.GetIndex();
		}
		if (is_deleted) {
			if (!initialized_sel) {
				result_sel.Initialize(STANDARD_VECTOR_SIZE);
				for (idx_t prev = 0; prev < i; prev++) {
					result_sel.set_index(prev, prev);
				}
				result_count = i;
				initialized_sel = true;
			}
			return;
		}
		if (initialized_sel) {
			result_sel.set_index(result_count++, i);
		}
	};

	if (is_unsigned) {
		auto positions = row_numbers.Values<uint64_t>();
		for (idx_t i = 0; i < count; i++) {
			check_row(i, positions[i].GetValue());
		}
	} else {
		auto positions = row_numbers.Values<int64_t>();
		for (idx_t i = 0; i < count; i++) {
			check_row(i, NumericCast<idx_t>(positions[i].GetValue()));
		}
	}
	return initialized_sel ? result_count : count;
}

DuckLakeFormatReader::DuckLakeFormatReader(ClientContext &context, DuckLakeFunctionInfo &read_info,
                                           const OpenFileInfo &info, string format_p)
    : BaseFileReader(info), read_info(read_info), format(std::move(format_p)) {
	file_scan = GetReadFunction(context, format);

	vector<Value> children;
	children.push_back(Value(file.path));
	named_argument_map_t named_params;
	vector<LogicalType> input_types;
	vector<Identifier> input_names;
	if (StringUtil::CIEquals(format, "json")) {
		named_params[Identifier("auto_detect")] = Value::BOOLEAN(true);
	}

	TableFunctionRef empty;
	BoundTableFunction bound_table_function(file_scan);
	TableFunctionBindInput bind_input(children, named_params, input_types, input_names, file_scan.function_info.get(),
	                                  nullptr, bound_table_function, empty);
	bind_data = file_scan.bind(context, bind_input, file_types, file_names);

	columns.reserve(file_names.size());
	for (idx_t i = 0; i < file_names.size(); i++) {
		auto col = MultiFileColumnDefinition::CreateNested(file_names[i], file_types[i]);
		if (StringUtil::CIEquals(file_names[i].GetIdentifierName(), "_ducklake_internal_row_id")) {
			col.identifier = Value::INTEGER(MultiFileReader::ROW_ID_FIELD_ID);
		} else if (StringUtil::CIEquals(file_names[i].GetIdentifierName(), "_ducklake_internal_snapshot_id")) {
			col.identifier = Value::INTEGER(MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID);
		}
		columns.push_back(std::move(col));
	}

	if (file_scan.get_virtual_columns) {
		auto virtual_cols = file_scan.get_virtual_columns(context, bind_data.get());
		auto row_num_entry = virtual_cols.find(MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER);
		if (row_num_entry != virtual_cols.end()) {
			supports_file_row_number = true;
			file_row_number_type = row_num_entry->second.type;
		}
		supports_empty_column = virtual_cols.count(COLUMN_IDENTIFIER_EMPTY) > 0;
	}
}

TableFunction DuckLakeFormatReader::GetReadFunction(ClientContext &context, const string &format) {
	auto &instance = DatabaseInstance::GetDatabase(context);
	ExtensionHelper::TryAutoLoadExtension(instance, format);
	ExtensionLoader loader(instance, "ducklake");
	auto &scan_entry = loader.GetTableFunction(Identifier("read_" + format));
	return *scan_entry.functions.functions[0];
}

void DuckLakeFormatReader::InitializeGlobalScan(ClientContext &context) {
	unordered_map<column_t, idx_t> file_pos_to_scan_idx;
	bool has_ordinal = false;

	for (const auto &column_id : column_indexes) {
		auto primary_idx = column_id.GetPrimaryIndex();
		if (primary_idx == COLUMN_IDENTIFIER_EMPTY || primary_idx >= columns.size()) {
			output_kinds.push_back(FormatReaderColumn::EMPTY);
			output_source_index.push_back(0);
			output_extract_paths.emplace_back();
			continue;
		}
		auto &col = columns[primary_idx];
		if (!col.identifier.IsNull() && col.identifier.type().id() == LogicalTypeId::INTEGER &&
		    IntegerValue::Get(col.identifier) == MultiFileReader::ORDINAL_FIELD_ID) {
			output_kinds.push_back(FormatReaderColumn::ORDINAL);
			output_source_index.push_back(0);
			output_extract_paths.emplace_back();
			has_ordinal = true;
			continue;
		}
		if (primary_idx >= file_types.size()) {
			throw InternalException("Format reader column index %llu out of bounds for %llu file columns", primary_idx,
			                        file_types.size());
		}

		vector<idx_t> extract_path;
		const ColumnIndex *cur_idx = &column_id;
		while (cur_idx->IsPushdownExtract() && cur_idx->HasChildren()) {
			cur_idx = &cur_idx->GetChildIndexes()[0];
			extract_path.push_back(cur_idx->GetPrimaryIndex());
		}

		idx_t scan_pos;
		auto scan_it = file_pos_to_scan_idx.find(primary_idx);
		if (scan_it != file_pos_to_scan_idx.end()) {
			scan_pos = scan_it->second;
		} else {
			scan_pos = scan_column_ids.size();
			file_pos_to_scan_idx.emplace(primary_idx, scan_pos);
			scan_column_ids.push_back(primary_idx);
			scan_types.push_back(file_types[primary_idx]);
		}
		output_kinds.push_back(FormatReaderColumn::FILE_COLUMN);
		output_source_index.push_back(scan_pos);
		output_extract_paths.push_back(std::move(extract_path));
	}

	bool needs_row_number = has_ordinal || deletion_filter != nullptr;
	use_virtual_file_row_number = needs_row_number && supports_file_row_number;
	if (use_virtual_file_row_number) {
		file_row_number_scan_idx = scan_column_ids.size();
		scan_column_ids.push_back(MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER);
		scan_types.push_back(file_row_number_type);
	}

	if (scan_column_ids.empty()) {
		if (supports_empty_column) {
			scan_column_ids.push_back(COLUMN_IDENTIFIER_EMPTY);
			scan_types.push_back(LogicalType::BOOLEAN);
		} else if (!file_types.empty()) {
			scan_column_ids.push_back(0);
			scan_types.push_back(file_types[0]);
		}
	}

	bool sequential_row_numbers = needs_row_number && !use_virtual_file_row_number;
	bool can_push_filters = file_scan.filter_pushdown && !sequential_row_numbers;
	if (can_push_filters && filters && filters->HasFilters()) {
		auto candidate_filters = make_uniq<TableFilterSet>();
		for (const auto &entry : *filters) {
			const auto &filter = entry.Filter();
			if (ExpressionFilter::IsRootOptionalFilter(filter)) {
				continue;
			}
			idx_t col_idx = entry.GetIndex().GetIndex();
			if (col_idx >= output_kinds.size() || output_kinds[col_idx] != FormatReaderColumn::FILE_COLUMN) {
				continue;
			}
			if (!output_extract_paths[col_idx].empty()) {
				continue;
			}
			if (expression_map.find(ProjectionIndex(col_idx)) != expression_map.end()) {
				continue;
			}
			auto &expr_filter = ExpressionFilter::GetExpressionFilter(filter, "DuckLakeFormatReader");
			candidate_filters->PushFilter(ProjectionIndex(output_source_index[col_idx]), expr_filter.Copy());
		}
		if (candidate_filters->HasFilters()) {
			pushed_filters = std::move(candidate_filters);
		}
	}

	TableFunctionInitInput init_input(bind_data.get(), scan_column_ids, vector<idx_t>(), pushed_filters.get());
	try {
		global_state = file_scan.init_global(context, init_input);
	} catch (const BinderException &) {
		if (!pushed_filters) {
			throw;
		}
		pushed_filters.reset();
		TableFunctionInitInput fallback_input(bind_data.get(), scan_column_ids, vector<idx_t>(), nullptr);
		global_state = file_scan.init_global(context, fallback_input);
	} catch (const NotImplementedException &) {
		if (!pushed_filters) {
			throw;
		}
		pushed_filters.reset();
		TableFunctionInitInput fallback_input(bind_data.get(), scan_column_ids, vector<idx_t>(), nullptr);
		global_state = file_scan.init_global(context, fallback_input);
	}

	if (sequential_row_numbers || !global_state) {
		max_threads = 1;
	} else {
		max_threads = MaxValue<idx_t>(1, global_state->MaxThreads());
	}
}

bool DuckLakeFormatReader::TryInitializeScan(ClientContext &context, GlobalTableFunctionState &gstate,
                                             LocalTableFunctionState &lstate) {
	lock_guard<mutex> guard(lock);
	if (scan_exhausted.load()) {
		return false;
	}
	if (!initialized_global) {
		InitializeGlobalScan(context);
		initialized_global = true;
	}
	if (claimed_threads >= max_threads || thread_states.count(&lstate)) {
		return false;
	}

	auto state = make_uniq<FormatReaderThreadState>();
	state->thread_context = make_uniq<ThreadContext>(context);
	state->execution_context = make_uniq<ExecutionContext>(context, *state->thread_context, nullptr);
	if (!scan_types.empty()) {
		state->scan_chunk.Initialize(context, scan_types);
	}

	for (const auto &entry : expression_map) {
		auto &expression = *entry.second.expression;
		state->expression_executors[entry.first.GetIndex()] = make_uniq<ExpressionExecutor>(context, expression);
	}
	if (filters) {
		for (const auto &entry : *filters) {
			const auto &filter = entry.Filter();
			if (ExpressionFilter::IsRootOptionalFilter(filter)) {
				continue;
			}
			state->filter_states.emplace_back(entry.GetIndex().GetIndex(),
			                                  TableFilterState::Initialize(context, filter));
		}
	}

	if (file_scan.init_local) {
		TableFunctionInitInput init_input(bind_data.get(), scan_column_ids, vector<idx_t>(), pushed_filters.get());
		state->local_state = file_scan.init_local(*state->execution_context, init_input, global_state.get());
		if (!state->local_state) {
			scan_exhausted.store(true);
			return false;
		}
	}

	thread_states.emplace(&lstate, std::move(state));
	claimed_threads++;
	return true;
}

bool DuckLakeFormatReader::TryEvaluateExpression(ClientContext &context, FormatReaderThreadState &thread_state,
                                                 idx_t output_col_idx, Vector &input_vector,
                                                 const LogicalType &input_type, idx_t count, Vector &output_vector) {
	auto expr_it = thread_state.expression_executors.find(output_col_idx);
	if (expr_it == thread_state.expression_executors.end()) {
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
	FormatReaderThreadState *thread_state = nullptr;
	{
		lock_guard<mutex> guard(lock);
		auto it = thread_states.find(&local_table_state);
		if (it == thread_states.end()) {
			return AsyncResult(SourceResultType::FINISHED);
		}
		thread_state = it->second.get();
	}

	while (true) {
		chunk.Reset();
		thread_state->scan_chunk.Reset();
		TableFunctionInput function_input(bind_data.get(), thread_state->local_state.get(), global_state.get());
		file_scan.function(context, function_input, thread_state->scan_chunk);
		idx_t scan_count = thread_state->scan_chunk.size();
		if (scan_count == 0) {
			scan_exhausted.store(true);
			lock_guard<mutex> guard(lock);
			thread_states.erase(&local_table_state);
			return AsyncResult(SourceResultType::FINISHED);
		}

		for (idx_t c = 0; c < output_kinds.size(); c++) {
			switch (output_kinds[c]) {
			case FormatReaderColumn::FILE_COLUMN: {
				auto &source = thread_state->scan_chunk.data[output_source_index[c]];
				if (!output_extract_paths[c].empty()) {
					if (thread_state->expression_executors.count(c)) {
						auto &target_type =
						    column_indexes[c].HasType() ? column_indexes[c].GetScanType() : chunk.data[c].GetType();
						Vector extracted(target_type);
						ExtractStructField(context, source, output_extract_paths[c], scan_count, extracted);
						TryEvaluateExpression(context, *thread_state, c, extracted, target_type, scan_count,
						                      chunk.data[c]);
					} else {
						ExtractStructField(context, source, output_extract_paths[c], scan_count, chunk.data[c]);
					}
					break;
				}
				if (TryEvaluateExpression(context, *thread_state, c, source, source.GetType(), scan_count,
				                          chunk.data[c])) {
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
				if (use_virtual_file_row_number) {
					auto &source = thread_state->scan_chunk.data[file_row_number_scan_idx.GetIndex()];
					if (thread_state->expression_executors.count(c)) {
						Vector ordinal_vector(LogicalType::BIGINT);
						if (source.GetType() != LogicalType::BIGINT) {
							VectorOperations::Cast(context, source, ordinal_vector, scan_count);
						} else {
							ordinal_vector.Reference(source);
						}
						TryEvaluateExpression(context, *thread_state, c, ordinal_vector, LogicalType::BIGINT,
						                      scan_count, chunk.data[c]);
					} else if (chunk.data[c].GetType() != source.GetType()) {
						VectorOperations::Cast(context, source, chunk.data[c], scan_count);
					} else {
						chunk.data[c].Reference(source);
					}
				} else {
					Vector ordinal_vector(LogicalType::BIGINT);
					VectorOperations::GenerateSequence(ordinal_vector, scan_count, file_row_number);
					if (TryEvaluateExpression(context, *thread_state, c, ordinal_vector, LogicalType::BIGINT,
					                          scan_count, chunk.data[c])) {
						break;
					}
					VectorOperations::Copy(ordinal_vector, chunk.data[c], scan_count, 0, 0);
				}
				break;
			}
			case FormatReaderColumn::EMPTY:
				break;
			}
		}
		chunk.SetChildCardinality(scan_count);

		idx_t approved_tuple_count = scan_count;
		if (deletion_filter || !thread_state->filter_states.empty()) {
			SelectionVector sel;
			if (deletion_filter) {
				if (use_virtual_file_row_number) {
					auto &row_numbers = thread_state->scan_chunk.data[file_row_number_scan_idx.GetIndex()];
					auto &ducklake_delete_filter = static_cast<const DuckLakeDeleteFilter &>(*deletion_filter);
					approved_tuple_count = FilterByRowNumbers(ducklake_delete_filter, row_numbers, scan_count, sel);
				} else {
					approved_tuple_count = deletion_filter->Filter(file_row_number, approved_tuple_count, sel);
				}
			}
			if (approved_tuple_count > 0) {
				for (auto &filter_entry : thread_state->filter_states) {
					approved_tuple_count = ColumnSegment::FilterSelection(
					    sel, chunk.data[filter_entry.first], *filter_entry.second, chunk.size(), approved_tuple_count);
					if (approved_tuple_count == 0) {
						break;
					}
				}
			}
			if (approved_tuple_count != chunk.size()) {
				chunk.Slice(sel, approved_tuple_count);
			}
		}

		if (!use_virtual_file_row_number) {
			file_row_number += NumericCast<int64_t>(scan_count);
		}
		if (approved_tuple_count > 0) {
			return AsyncResult(SourceResultType::HAVE_MORE_OUTPUT);
		}
	}
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
