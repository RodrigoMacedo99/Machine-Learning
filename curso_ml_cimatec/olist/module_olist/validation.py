"""Validate raw Olist datasets against their pandera schemas and report the
most common data-quality errors found across all tables."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from loguru import logger
import typer

from module_olist.config import RAW_DATA_DIR, REPORTS_DIR
from module_olist.schemas import SCHEMA_REGISTRY

app = typer.Typer()


def validate_table(csv_name: str, schema: type[pa.DataFrameModel], input_dir: Path) -> pd.DataFrame:
    """Validate a single raw CSV against its schema, returning its failure cases (empty if valid)."""
    df = pd.read_csv(input_dir / csv_name)
    try:
        schema.validate(df, lazy=True)
        logger.success(f"{csv_name}: OK ({len(df)} linhas, sem erros)")
        return pd.DataFrame()
    except pa.errors.SchemaErrors as err:
        failures = err.failure_cases.copy()
        failures.insert(0, "table", csv_name)
        logger.warning(f"{csv_name}: {len(failures)} erro(s) em {len(df)} linhas")
        return failures


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_path: Path = REPORTS_DIR / "pandera_validation_failures.csv",
    summary_path: Path = REPORTS_DIR / "pandera_validation_summary.csv",
):
    """Run pandera validation across every table in SCHEMA_REGISTRY and write two reports:
    the full list of failure cases and a summary of the most common error types."""
    logger.info("Validando datasets brutos da Olist com pandera...")

    all_failures = [
        validate_table(csv_name, schema, input_dir) for csv_name, schema in SCHEMA_REGISTRY.items()
    ]
    failures_df = pd.concat(all_failures, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_df.to_csv(output_path, index=False)
    logger.info(f"Falhas detalhadas salvas em {output_path} ({len(failures_df)} linhas)")

    if failures_df.empty:
        logger.success("Nenhum erro encontrado em nenhuma tabela.")
        pd.DataFrame(columns=["table", "column", "check", "ocorrencias"]).to_csv(
            summary_path, index=False
        )
        return

    summary = (
        failures_df.groupby(["table", "column", "check"], dropna=False)
        .size()
        .reset_index(name="ocorrencias")
        .sort_values("ocorrencias", ascending=False)
    )
    summary.to_csv(summary_path, index=False)
    logger.info(f"Resumo dos erros mais comuns salvo em {summary_path}")

    logger.info("Top 10 erros mais comuns:")
    for _, row in summary.head(10).iterrows():
        logger.info(f"  {row['table']} | {row['column']} | {row['check']} -> {row['ocorrencias']}")


if __name__ == "__main__":
    app()
