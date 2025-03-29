from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import logging


def extract_metadata(data_file: str, logger) -> Dict[str, Any]:
    """
    Extract metadata from the data file.
    
    Args:
        data_file: Path to the data file
        logger: Logger instance
        
    Returns:
        Dict: Extracted metadata
    """
    logger.info(f"Extracting metadata from {data_file}")

    try:
        file_path = Path(data_file)
        basic_metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "source_path": str(file_path.absolute())
        }

        # Extract additional metadata based on file type
        if file_path.suffix.lower() == '.pdf':
            pdf_metadata = _extract_pdf_metadata(file_path, logger)
            if pdf_metadata:
                basic_metadata.update(pdf_metadata)

        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            image_metadata = _extract_image_metadata(file_path, logger)
            if image_metadata:
                basic_metadata.update(image_metadata)

        elif file_path.suffix.lower() == '.html':
            html_metadata = _extract_html_metadata(file_path, logger)
            if html_metadata:
                basic_metadata.update(html_metadata)

        elif file_path.suffix.lower() == '.csv':
            csv_metadata = _extract_csv_metadata(file_path, logger)
            if csv_metadata:
                basic_metadata.update(csv_metadata)

        logger.debug(f"Extracted metadata: {json.dumps(basic_metadata, indent=2)}")
        return basic_metadata

    except Exception as e:
        logger.error(f"Error in extract_metadata: {str(e)}", exc_info=True)
        return {
            "filename": data_file,
            "error": str(e)
        }


def _extract_pdf_metadata(file_path: Path, logger) -> Dict[str, Any]:
    """Extract metadata from PDF files."""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            if pdf.metadata:
                return {
                    "title": pdf.metadata.get('/Title', ''),
                    "author": pdf.metadata.get('/Author', ''),
                    "creator": pdf.metadata.get('/Creator', ''),
                    "producer": pdf.metadata.get('/Producer', ''),
                    "creation_date": pdf.metadata.get('/CreationDate', ''),
                    "modification_date": pdf.metadata.get('/ModDate', ''),
                    "page_count": len(pdf.pages)
                }
        return {}
    except ImportError:
        logger.warning("PyPDF2 not installed, skipping PDF metadata extraction")
        return {}
    except Exception as e:
        logger.warning(f"Error extracting PDF metadata: {str(e)}")
        return {}


def _extract_image_metadata(file_path: Path, logger) -> Dict[str, Any]:
    """Extract metadata from image files."""
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            return {
                "image_format": img.format,
                "image_size": img.size,
                "image_mode": img.mode,
                "image_info": {k: str(v) for k, v in img.info.items()}  # Convert values to strings to ensure JSON serialization
            }
    except ImportError:
        logger.warning("Pillow not installed, skipping image metadata extraction")
        return {}
    except Exception as e:
        logger.warning(f"Error extracting image metadata: {str(e)}")
        return {}


def _extract_html_metadata(file_path: Path, logger) -> Dict[str, Any]:
    """Extract metadata from HTML files."""
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')
                if name and content:
                    meta_tags[name] = content

            return {
                "title": soup.title.string if soup.title else '',
                "meta_tags": meta_tags,
                "has_article": bool(soup.find('article')),
                "has_main": bool(soup.find('main')),
                "num_headers": len(soup.find_all(['h1', 'h2', 'h3'])),
                "has_tables": bool(soup.find_all('table'))
            }
    except ImportError:
        logger.warning("BeautifulSoup4 not installed, skipping HTML metadata extraction")
        return {}
    except Exception as e:
        logger.warning(f"Error extracting HTML metadata: {str(e)}")
        return {}


def _extract_csv_metadata(file_path: Path, logger) -> Dict[str, Any]:
    """Extract metadata from CSV files."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "has_nulls": df.isnull().any().any()
        }
    except Exception as e:
        logger.warning(f"Error extracting CSV metadata: {str(e)}")
        return {}