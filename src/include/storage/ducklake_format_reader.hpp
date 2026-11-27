//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_format_reader.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/multi_file/base_file_reader.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/table_filter_state.hpp"

namespace duckdb {
struct DuckLakeFunctionInfo;

//! How an output column is produced.
enum class FormatReaderColumn : uint8_t {
	FILE_COLUMN, //! read from the file (see output_source_index)
	ORDINAL,     //! file_row_number, emitted by the reader
	EMPTY        //! placeholder column not populated from the file
};

//! Per-worker state for scanning a format reader concurrently.
struct FormatReaderThreadState {
	unique_ptr<ThreadContext> thread_context;
	unique_ptr<ExecutionContext> execution_context;
	unique_ptr<LocalTableFunctionState> local_state;
	DataChunk scan_chunk;
	unordered_map<idx_t, unique_ptr<ExpressionExecutor>> expression_executors;
	vector<pair<idx_t, unique_ptr<TableFilterState>>> filter_states;
};

//! Reads a single non-parquet DuckLake data file ("vortex", "json", ...) by driving its read_<format>
//! table function and adapting the chunks to DuckLake's MultiFileReader contract.
class DuckLakeFormatReader : public BaseFileReader {
public:
	DuckLakeFormatReader(ClientContext &context, DuckLakeFunctionInfo &read_info, const OpenFileInfo &info,
	                     string format);

public:
	bool TryInitializeScan(ClientContext &context, GlobalTableFunctionState &gstate,
	                       LocalTableFunctionState &lstate) override;
	AsyncResult Scan(ClientContext &context, GlobalTableFunctionState &global_state,
	                 LocalTableFunctionState &local_state, DataChunk &chunk) override;
	string GetReaderType() const override;
	void AddVirtualColumn(column_t virtual_column_id) override;

private:
	static TableFunction GetReadFunction(ClientContext &context, const string &format);
	void InitializeGlobalScan(ClientContext &context);
	bool TryEvaluateExpression(ClientContext &context, FormatReaderThreadState &thread_state, idx_t output_col_idx,
	                           Vector &input_vector, const LogicalType &input_type, idx_t count, Vector &output_vector);

	mutex lock;
	DuckLakeFunctionInfo &read_info;
	string format;

	TableFunction file_scan;
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> file_types;
	vector<Identifier> file_names;
	bool supports_file_row_number = false;
	bool supports_empty_column = false;
	LogicalType file_row_number_type = LogicalType::UBIGINT;

	bool initialized_global = false;
	bool use_virtual_file_row_number = false;
	idx_t max_threads = 1;
	idx_t claimed_threads = 0;
	atomic<bool> scan_exhausted {false};
	int64_t file_row_number = 0;

	vector<column_t> scan_column_ids;
	vector<LogicalType> scan_types;
	optional_idx file_row_number_scan_idx;
	vector<FormatReaderColumn> output_kinds;
	vector<idx_t> output_source_index;
	vector<vector<idx_t>> output_extract_paths;

	unique_ptr<TableFilterSet> pushed_filters;
	unique_ptr<GlobalTableFunctionState> global_state;
	unordered_map<LocalTableFunctionState *, unique_ptr<FormatReaderThreadState>> thread_states;
};

} // namespace duckdb
