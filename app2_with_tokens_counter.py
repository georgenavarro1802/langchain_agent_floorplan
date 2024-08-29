import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def calculate_image_tokens(image_path, detail="auto"):
    with Image.open(image_path) as img:
        width, height = img.size

    if detail == "low" or (detail == "auto" and max(width, height) <= 512):
        return 85

    # Scale image to fit within 2048x2048
    scale = min(2048 / max(width, height), 1)
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    # Scale such that the shortest side is 768px
    scale = max(768 / min(scaled_width, scaled_height), 1)
    final_width, final_height = int(scaled_width * scale), int(scaled_height * scale)

    # Count 512px tiles
    tiles = ((final_width + 511) // 512) * ((final_height + 511) // 512)

    return 85 + (170 * tiles)


def calculate_cost(prompt_tokens, completion_tokens, image_tokens):
    # GPT-4o pricing as of the last update
    input_price_per_1k = 0.01
    output_price_per_1k = 0.03

    input_cost = (prompt_tokens + image_tokens) * input_price_per_1k / 1000
    output_cost = completion_tokens * output_price_per_1k / 1000

    return input_cost + output_cost


def analyze_floor_plan(image_path):
    base64_image = encode_image(image_path)
    # image_tokens = calculate_image_tokens(image_path)

    messages = [
        {
            "role": "system",
            "content": """
            You are an AI specialized in analyzing data center floor plans. Your task is to examine the provided image 
            and generate a structured JSON representation of the data center layout. Follow these guidelines strictly:

            1. The JSON should be a flat array of objects, each representing either a row or a rack.
            2. Identify all rows in the layout, regardless of their orientation (vertical or horizontal).
            3. For rack dimensions, use exactly:
               - For vertical rows, racks: width = 90 pixels, height = 60 pixels
               - For horizontal rows, racks: width = 60 pixels, height = 90 pixels
            4. Each row object must have this structure:
               {
                 "id": "r{row_number}",
                 "label": "Row {row_number}",
                 "position": {"x": {calculated_x}, "y": {calculated_y}},
                 "height": {calculated_height},
                 "width": {calculated_width},
                 "connectable": false,
                 "selectable": false,
                 "class": "row"
               }
            5. Each rack object must have this structure:
               {
                 "id": "r{row_number}-{rack_number}",
                 "label": "{rack_name_from_image}",
                 "position": {"x": {calculated_x}, "y": {calculated_y}},
                 "width": {rack_width},
                 "height": {rack_height},
                 "parentNode": "r{row_number}",
                 "connectable": false,
                 "selectable": false,
                 "class": "rack",
                 "type": "{IT_or_REF}"
               }
            6. Calculate row dimensions:
               - Height/width must cover all racks plus 20 pixels padding (10 on each side)
            7. Position rows to avoid overlapping:
               - Ensure 50 pixels spacing between rows
            8. Position each rack relative to its parent row:
               - Coordinates must be relative to the top-left corner of the parent row
               - For vertical rows:
                 * x = 10 for all racks
                 * y starts at 20 for the first rack, increase by (rack height + 1) for each subsequent rack
               - For horizontal rows:
                 * x starts at 20 for the first rack, increase by (rack width + 1) for each subsequent rack
                 * y = 10 for all racks
            9. Mark cooling units (AC, AHU) as type "REF". All others are type "IT".
            10. Use visible identifiers in the image for rack labels.
            11. All measurements must be in pixels and integers.
            12. Ensure all racks are within their row boundaries.

            Analyze the image carefully and create the JSON structure based on these precise instructions.
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this data center floor plan and provide the JSON structure as described in the instructions. Respond only with the JSON array, no additional text."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    print("Processing the floor plan...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4096,
        temperature=0.2  # Lower temperature for more consistent results
    )

    try:
        json_structure = json.loads(response.choices[0].message.content)
        print("Floor plan analysis complete. Here's the processed JSON structure:")
        print(json.dumps(json_structure, indent=2))

        prompt_tokens = response.usage.prompt_tokens
        # completion_tokens = response.usage.completion_tokens
        # total_cost = calculate_cost(prompt_tokens, completion_tokens, image_tokens)
        #
        # print(f"\nToken Usage:")
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Image tokens: {image_tokens}")
        # print(f"Estimated cost: ${total_cost:.4f}")

        return json_structure
    except json.JSONDecodeError as je:
        print('JSONDecodeError:', str(je))
        print('Raw response:', response.choices[0].message.content)
        return None


if __name__ == "__main__":
    image_path = "layout_9.jpg"
    result = analyze_floor_plan(image_path)
    if result:
        # You can do further processing with the result here if needed
        pass
