"""Run from the project root with the aggregator running: python test_mcp_tools.py."""

import asyncio
import shutil
import uuid

from agent.mcp_client import aggregator_tools
from config.settings import DATA, UPLOADS


async def main():
    async with aggregator_tools() as tools:
        names = {tool.name: tool for tool in tools}
        print("Tools:", ", ".join(sorted(names)))
        for filename in ("titanic.csv", "ProjectPlan.xlsx"):
            dataset_id = uuid.uuid4().hex
            upload = UPLOADS / f"{dataset_id}_{filename}"
            shutil.copyfile(DATA / filename, upload)
            try:
                arguments = {"dataset_id": dataset_id}
                schema = await names["pandas_get_dataset_info"].ainvoke(arguments)
                print(filename, schema)
                if filename.endswith(".csv"):
                    arguments.update(group_columns=["pclass"], value_column="fare", operation="mean")
                else:
                    arguments.update(
                        sheet="DevPlan", table="Table 1",
                        group_columns=["Financial / Non-Financial"], operation="size",
                    )
                print(await names["pandas_groupby_data"].ainvoke(arguments))
            finally:
                upload.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
