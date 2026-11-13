"""Validated analytics over a selected table in the uploaded dataset."""
from pathlib import Path
import pandas as pd
from rag_mcp.excel_ingestion import load_excel_workbook
from config.settings import UPLOADS
import sys
from functools import lru_cache

def _tables(dataset_id):
    if (
        not isinstance(dataset_id, str)
        or not dataset_id.isalnum()
        or len(dataset_id) != 32
    ):
        raise ValueError("Invalid dataset identifier")

    paths = list(UPLOADS.glob(f"{dataset_id}_*"))

    if len(paths) != 1:
        raise ValueError("Dataset not found")

    path = paths[0]

    return _load_tables_cached(
        str(path),
        path.stat().st_mtime_ns,
        path.stat().st_size
    )


@lru_cache(maxsize=10)
def _load_tables_cached(file_path, modified_time, file_size):
    path = Path(file_path)

    print(
        f"[PANDAS] Loading dataset: {path.name}",
        file=sys.stderr,
        flush=True
    )

    if path.suffix.lower() == ".csv":
        return [
            {
                "dataframe": pd.read_csv(path),
                "metadata": {
                    "sheet": "CSV",
                    "table": "Table 1",
                    "orientation": "horizontal"
                }
            }
        ]

    tables = load_excel_workbook(str(path))["tables"]
    if not tables:
        raise ValueError("No data tables found in the workbook")
    return tables


def _select(dataset_id,sheet=None,table=None):
    tables=_tables(dataset_id)
    if sheet is not None: tables=[t for t in tables if t['metadata']['sheet']==sheet]
    if table is not None: tables=[t for t in tables if str(t['metadata']['table'])==str(table)]
    if not tables: raise ValueError('No matching table; call get_dataset_info to inspect sheets and tables')
    if len(tables)>1: raise ValueError('Multiple tables match; specify sheet and table from get_dataset_info')
    return tables[0]['dataframe'].copy()

def _safe(x):
    if isinstance(x,dict): return {str(k):_safe(v) for k,v in x.items()}
    if isinstance(x,list): return [_safe(v) for v in x]
    if isinstance(x,tuple): return [_safe(v) for v in x]
    if isinstance(x,(pd.Timestamp,)): return x.isoformat()
    try:
        if pd.isna(x): return None
    except (TypeError,ValueError): pass
    if hasattr(x,'item'): return x.item()
    return x

def get_dataset_info(dataset_id):
    return {'dataset_id':dataset_id,'tables':[{'sheet':t['metadata']['sheet'],
        'table':str(t['metadata']['table']),'rows':len(t['dataframe']),
        'column_count':len(t['dataframe'].columns),
        **({'project': t['metadata']['project']} if t['metadata'].get('project') else {}),
        'columns':list(map(str,t['dataframe'].columns)),
        'dtypes':{str(c):str(d) for c,d in t['dataframe'].dtypes.items()}}
        for t in _tables(dataset_id)]}

def _col(df,name):
    if name not in df: raise ValueError(f'Unknown column: {name}. Available: {list(df.columns)}')

def _filtered(df,filters):
    for f in filters or []:
        c,op,value=f['column'],f['operator'],f.get('value')
        _col(df,c)
        s=df[c]
        if op=='eq': mask=s.eq(value)
        elif op=='ne': mask=s.ne(value)
        elif op in ('gt','gte','lt','lte'):
            numeric=pd.to_numeric(s,errors='coerce'); v=float(value)
            mask={'gt':numeric.gt,'gte':numeric.ge,'lt':numeric.lt,'lte':numeric.le}[op](v)
        elif op=='contains': mask=s.astype('string').str.contains(str(value),case=False,regex=False,na=False)
        elif op=='in':
            if not isinstance(value,list): raise ValueError('in requires a list')
            mask=s.isin(value)
        elif op=='is_null': mask=s.isna()
        elif op=='not_null': mask=s.notna()
        else: raise ValueError(f'Unsupported operator: {op}')
        df=df[mask.fillna(False)]
    return df

def get_column_summary(dataset_id,column,sheet=None,table=None):
    df=_select(dataset_id,sheet,table);_col(df,column);s=df[column]
    out={'column':column,'dtype':str(s.dtype),'count':int(s.count()),'missing':int(s.isna().sum()),'unique':int(s.nunique())}
    if pd.api.types.is_numeric_dtype(s): out.update({k:_safe(getattr(s,k)()) for k in ('mean','sum','min','max','median')})
    else: out['top_values']={str(k):int(v) for k,v in s.value_counts().head(10).items()}
    return out

def aggregate_data(dataset_id,column,operation,filters=None,sheet=None,table=None):
    df=_filtered(_select(dataset_id,sheet,table),filters);_col(df,column)
    if operation not in ('sum','mean','median','min','max','count','nunique'):
        raise ValueError('Unsupported aggregation')
    s=df[column]
    if operation in ('sum','mean','median'): s=pd.to_numeric(s,errors='coerce')
    value = s.sum(min_count=1) if operation == 'sum' else getattr(s, operation)()
    return _safe({'operation':operation,'column':column,'value':value, 'rows':len(df)})

def groupby_data(dataset_id, group_columns, value_column=None, operation="count",
                 filters=None, sheet=None, table=None, limit=100):
    df = _filtered(_select(dataset_id, sheet, table), filters)
    if isinstance(group_columns, str):
        group_columns = [group_columns]
    if not group_columns or len(set(group_columns)) != len(group_columns):
        raise ValueError("Specify one or more distinct group columns")
    for column in group_columns:
        _col(df, column)

    groups = df.groupby(group_columns, dropna=False, observed=True)
    if operation == "size":
        counts = groups.size()
        result_column = "row_count"
    else:
        if operation not in ("sum", "mean", "median", "min", "max", "count", "nunique"):
            raise ValueError("Unsupported aggregation")
        _col(df, value_column)
        if operation in ("sum", "mean", "median"):
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        values = groups[value_column]
        counts = values.sum(min_count=1) if operation == "sum" else values.agg(operation)
        result_column = f"{value_column}_{operation}" if value_column in group_columns else value_column

    while result_column in group_columns:
        result_column = f"{result_column}_result"
    result = counts.reset_index(name=result_column)
    rows = result.head(max(1, min(int(limit), 500))).to_dict("records")
    return _safe({"groups": len(result), "rows": rows, "truncated": len(rows) < len(result)})


def filter_dataset(dataset_id,filters,sheet=None,table=None,limit=100):
    df=_filtered(_select(dataset_id,sheet,table),filters)
    return _safe({'matched':len(df),'rows':df.head(max(1,min(int(limit),500))).to_dict('records')})
