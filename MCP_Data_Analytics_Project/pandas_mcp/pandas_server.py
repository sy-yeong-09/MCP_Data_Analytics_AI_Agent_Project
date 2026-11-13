from fastmcp import FastMCP
from pandas_mcp import analytics as a
mcp=FastMCP('Pandas Analytics')
@mcp.tool
def get_dataset_info(dataset_id:str)->dict:
    """List detected worksheets, tables, column names and row counts."""
    return a.get_dataset_info(dataset_id)
@mcp.tool
def get_column_summary(dataset_id:str,column:str,sheet:str|None=None,table:str|None=None)->dict:
    """Describe one column; specify sheet/table when workbook has multiple tables."""
    return a.get_column_summary(dataset_id,column,sheet,table)
@mcp.tool
def aggregate_data(dataset_id:str,column:str,operation:str,filters:list[dict]|None=None,sheet:str|None=None,table:str|None=None)->dict:
    """Calculate sum, mean, median, min, max, count, nunique with optional filters."""
    return a.aggregate_data(dataset_id,column,operation,filters,sheet,table)
@mcp.tool
def groupby_data(dataset_id:str,group_columns:list[str],value_column:str|None=None,operation:str='count',filters:list[dict]|None=None,sheet:str|None=None,table:str|None=None,limit:int=100)->dict:
    """Group by exact column names. Use size to count rows without a value column. Count counts nonblank values."""
    return a.groupby_data(dataset_id,group_columns,value_column,operation,filters,sheet,table,limit)
@mcp.tool
def filter_dataset(dataset_id:str,filters:list[dict],sheet:str|None=None,table:str|None=None,limit:int=100)->dict:
    """Filter using eq/ne/gt/gte/lt/lte/contains/in/is_null/not_null."""
    return a.filter_dataset(dataset_id,filters,sheet,table,limit)
if __name__=='__main__':mcp.run(transport='stdio', show_banner=False)
