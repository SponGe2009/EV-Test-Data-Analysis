import json


def serialize_template_details(template_details, output_file_path):
    # Convert the template details into a JSON string
    with open(output_file_path, 'w') as file:
        # 'default=str' to handle any non-serializable objects like datetime
        json.dump(template_details, file, indent=4, default=str)


def deserialize_template_details(input_file_path):
    # Read the template details from a JSON file and convert them back into a dictionary
    # While ensuring that numeric values are correctly converted
    with open(input_file_path, 'r') as file:
        template_details = json.load(file)

    # Post-processing to ensure correct data types
    for sheet, details in template_details.items():
        # Ensure row heights and column widths are the correct type
        details['row_heights'] = {
            int(k): v for k, v in details['row_heights'].items() if v is not None}
        details['column_widths'] = {
            k: float(v) for k, v in details['column_widths'].items() if v is not None}

        # Additional type corrections can be added here as needed

    return template_details

# Usage example (remember to uncomment when ready to use):
# serialize_template_details(template_details, '/path/to/template_details.json')
# loaded_template_details = deserialize_template_details('/path/to/template_details.json')
