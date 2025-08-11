# PDF/Image Text Extraction Project

This project provides an AI-powered solution for extracting text content from PDF files and images using a vision-language model (VLM). It processes documents in batches and saves results to text files.

## Features

- Extract text from PDF files (batch processing)
- Extract text from single images
- Multiple prompt modes for different extraction needs
- Configurable batch processing for large PDFs
- Results saved to organized text files

## Setup Instructions

### 1. Create Conda Environment

First, create a new conda environment with Python:

```bash
conda create -n pdf-extraction python=3.8
conda activate pdf-extraction
```

### 2. Install Required Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

The required dependencies include:

- PyMuPDF (for PDF processing)
- fitz (PDF library)
- openai (API client)
- python-dotenv (environment variables)
- requests (HTTP requests)
- tqdm (progress bars)
- Pillow (image processing)

### 3. Configure Input File

Open [`inference.py`](inference.py) and modify line 175 to specify your input file:

```python
# Line 175 in inference.py
input_path = "path/to/your/file.pdf"  # or "path/to/your/image.jpg"
```

**Examples:**

- For PDF: `input_path = "demo/10-pages.pdf"`
- For Image: `input_path = "demo/demo_image1.jpg"`

### 4. Create Environment Configuration

Create a [`.env`](.env) file in the project root directory and add your API configuration:

```env
IP_ADDRESS=https://your-api-endpoint.com
API_KEY=your_api_key_here
```

### 5. Run the Application

Execute the main script:

```bash
python inference.py
```

## Usage

### Processing PDFs

When you specify a PDF file, the application will:

- Load all pages from the PDF
- Process them in batches (default: 5 pages per batch)
- Save results to the [`results/`](results/) directory
- Create separate files for each batch

### Processing Images

When you specify an image file, the application will:

- Process the single image
- Save the result to a single text file in the [`results/`](results/) directory

### Configuration Options

You can modify these parameters in the [`main()`](inference.py) function:

```python
batch_size = 5      # Number of pages per batch for PDF processing
temperature = 0.1   # AI model temperature (creativity level)
top_p = 0.9        # AI model top-p parameter
result_dir = "results"  # Output directory
```

### Available Prompt Modes

The application supports different extraction modes via [`dict_promptmode_to_prompt`](inference.py):

- `"prompt_ocr"`: Basic text extraction
- `"prompt_layout_all_en"`: Full layout analysis with bounding boxes
- `"prompt_layout_only_en"`: Layout detection only
- `"prompt_grounding_ocr"`: Text extraction from specific regions

## Output

Results are saved in the [`results/`](results/) directory with the following naming convention:

- **PDF files**: `filename_batch_XXX_pages_Y_Z.txt`
- **Image files**: `filename_result.txt`

## Project Structure

```
.
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── inference.py           # Main application script
├── pdf_parse.py          # PDF processing utilities
├── demo/                 # Sample input files
│   ├── 10-pages.pdf
│   ├── demo_image1.jpg
│   └── page1-kt.pdf
└── results/              # Output directory
    ├── 10-pages_batch_001_pages_1_5.txt
    ├── 10-pages_batch_002_pages_6_10.txt
    └── demo_image1_result.txt
```

## Troubleshooting

1. **API Connection Issues**: Verify your IP_ADDRESS and API_KEY in the [`.env`](.env) file
2. **File Not Found**: Ensure the input file path is correct and the file exists
3. **Memory Issues**: Reduce batch_size for large PDFs
4. **Permission Errors**: Ensure write permissions for the results directory

## Example Usage

```bash
# Activate environment
conda activate pdf-extraction

# Process a PDF file
# (Edit line 175 in inference.py first)
python inference.py

# Results will be saved in results/ directory
```

## Notes

- The application uses a vision-language model for text extraction
- Processing time depends on document size and API response time
- Results are saved incrementally during batch processing
- The application handles both PDF and image formats automatically
