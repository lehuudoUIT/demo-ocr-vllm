import json
import io
import base64
import math
from PIL import Image
from dotenv import load_dotenv
import requests
from openai import OpenAI
import os
from io import BytesIO
from pdf_parse import load_images_from_pdf
# Load environment variables
load_dotenv()

dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_layout_only_en: parse ocr text except the Page-header and Page-footer
    "prompt_ocr": """Extract the text content from this image.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
}


def PILimage_to_base64(image, format='PNG'):
    buffered = BytesIO()
    image.save(buffered, format=format)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{base64_str}"

def inference_with_vllm(
        image,
        prompt, 
        ip="localhost",
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='dotsocr-model',
        ):
    # os.environ.get("API_KEY", "0")
    addr = f"{ip}/v1"
    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=addr, timeout=1000000)
    messages = []

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image, format='JPEG')},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
            ],
        }
    )
  
    try:
        response = client.chat.completions.create(
            messages=messages, 
            model=model_name, 
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None

def process_pdf_batch(input_path, prompt, batch_size=5, result_dir="results", ip="34.124.226.211", temperature=0.1, top_p=0.9):
    """
    Process PDF pages in batches and save results to text files.
    
    Args:
        input_path (str): Path to the PDF file
        prompt (str): Prompt to use for inference
        batch_size (int): Number of pages to process in each batch
        result_dir (str): Directory to save results
        ip (str): IP address of the inference server
        temperature (float): Temperature parameter for inference
        top_p (float): Top-p parameter for inference
    """
    filename, file_ext = os.path.splitext(os.path.basename(input_path))
    
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Load all images from PDF
    images = load_images_from_pdf(input_path, dpi=200)
    total_pages = len(images)
    
    print(f"Total pages: {total_pages}")
    print(f"Batch size: {batch_size}")
    print(f"Results will be saved to: {result_dir}")
    
    # Process pages in batches
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_num = (batch_start // batch_size) + 1
        
        print(f"\nProcessing batch {batch_num}: pages {batch_start + 1} to {batch_end}")
        
        # Create batch result file
        batch_filename = f"{filename}_batch_{batch_num:03d}_pages_{batch_start + 1}_{batch_end}.txt"
        batch_filepath = os.path.join(result_dir, batch_filename)
        
        batch_results = []
        
        # Process each page in the current batch
        for i in range(batch_start, batch_end):
            page_num = i + 1
            print(f"  Processing page {page_num}...")
            
            try:
                response = inference_with_vllm(
                    images[i],
                    prompt, 
                    ip=ip,
                    temperature=temperature,
                    top_p=top_p,
                )
                
                if response:
                    page_result = f"=== PAGE {page_num} ===\n{response}\n\n"
                    batch_results.append(page_result)
                    print(f"  Page {page_num} processed successfully")
                else:
                    error_msg = f"=== PAGE {page_num} ===\nERROR: Failed to get response\n\n"
                    batch_results.append(error_msg)
                    print(f"  Page {page_num} failed")
                    
            except Exception as e:
                error_msg = f"=== PAGE {page_num} ===\nERROR: {str(e)}\n\n"
                batch_results.append(error_msg)
                print(f"  Page {page_num} error: {e}")
        
        # Save batch results to file
        try:
            with open(batch_filepath, 'w', encoding='utf-8') as f:
                f.writelines(batch_results)
            print(f"  Batch {batch_num} results saved to: {batch_filename}")
        except Exception as e:
            print(f"  Error saving batch {batch_num}: {e}")
    
    print(f"\nProcessing complete! All results saved to: {result_dir}")

def main():
    input_path = "demo/demo_image1.jpg"
    prompt = dict_promptmode_to_prompt["prompt_ocr"]
    filename, file_ext = os.path.splitext(os.path.basename(input_path))
    
    # Configuration
    batch_size = 5  # Process 5 pages at a time
    result_dir = "results"
    ip = os.getenv("IP_ADDRESS")
    temperature = 0.1
    top_p = 0.9
    
    if file_ext == ".pdf":
        process_pdf_batch(
            input_path=input_path,
            prompt=prompt,
            batch_size=batch_size,
            result_dir=result_dir,
            ip=ip,
            temperature=temperature,
            top_p=top_p
        )
    else:
        # Handle single image files
        image = Image.open(input_path)
        response = inference_with_vllm(
            image,
            prompt, 
            ip=ip,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Save single image result
        os.makedirs(result_dir, exist_ok=True)
        result_filename = f"{filename}_result.txt"
        result_filepath = os.path.join(result_dir, result_filename)
        
        with open(result_filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== {filename} ===\n{response}\n")
        
        print(f"Result saved to: {result_filename}")


if __name__ == "__main__":
    main()