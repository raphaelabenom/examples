import logging
import zipfile
from pathlib import Path

import httpx
import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FetchData:
    """Download, extract, and organize educational data files from INEP."""

    def __init__(self):
        self.DATA = {
            "CENSO_ESCOLAR": "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2024.zip",
            "ENEM": "https://download.inep.gov.br/microdados/microdados_enem_2024.zip",
        }

    def get_data(self, output_dir: str = ".") -> list[Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        downloaded_files = []

        for name, url in self.DATA.items():
            filepath = output_path / f"{name.lower()}.zip"

            try:
                with httpx.stream(
                    "GET", url, timeout=300.0, follow_redirects=True
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))

                    with (
                        open(filepath, "wb") as file,
                        tqdm.tqdm(
                            total=total, unit="B", unit_scale=True, desc=name
                        ) as pbar,
                    ):
                        for data in response.iter_bytes():
                            file.write(data)
                            pbar.update(len(data))

                logger.info(f"{name} baixado com sucesso!")
                downloaded_files.append(filepath)

            except httpx.HTTPError as e:
                logger.error(f"Erro HTTP ao baixar {name}: {e}")
            except Exception as e:
                logger.error(f"Erro inesperado com {name}: {e}")

        return downloaded_files

    def unzip_data(self, input_dir: str = ".", output_dir: str = ".") -> list[Path]:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        zip_files = list(input_path.glob("*.zip"))

        if not zip_files:
            logger.warning("Nenhum arquivo .zip encontrado!")
            return []

        extracted_dirs = []

        for filepath in zip_files:
            try:
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(output_path)
                logger.info(f"{filepath.name} descompactado com sucesso!")
                extracted_dirs.append(output_path)
            except zipfile.BadZipFile:
                logger.error(f"Arquivo corrompido: {filepath.name}")
            except Exception as e:
                logger.error(f"Erro ao descompactar {filepath.name}: {e}")

        return extracted_dirs

    def find_files(
        self, search_path_files: str, files_names: list[str], move_output_dir: str
    ) -> list[Path]:
        input_path = Path(search_path_files)
        output_path = Path(move_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        moved_files = []

        for file_name in files_names:
            found = False
            for filepath in input_path.glob(f"**/{file_name}"):
                destination = output_path / file_name
                filepath.rename(destination)
                moved_files.append(destination)
                found = True
                logger.info(f"Movido: {file_name}")

            if not found:
                logger.warning(f"Arquivo não encontrado: {file_name}")

        return moved_files


if __name__ == "__main__":
    ingestion = FetchData()

    # Download data
    # ingestion.get_data("./data/raw")

    # Extract data
    # ingestion.unzip_data("./data/raw", "./data/processed")

    # Move specific files
    moved = ingestion.find_files(
        "./data/processed",
        ["RESULTADOS_2024.csv", "microdados_ed_basica_2024.csv"],
        "./data/bronze",
    )
    logger.info(f"Total de arquivos movidos: {len(moved)}")
