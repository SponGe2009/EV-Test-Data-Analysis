import json
import re


import json
import re

# Function to convert column letter(s) to number


def column_to_number(col_str):
    num = 0
    for c in col_str.upper():
        num = num * 26 + (ord(c) - ord('A') + 1)
    return num

# Function to convert number back to column letter(s)


def number_to_column(n):
    col_str = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        col_str = chr(65 + remainder) + col_str
    return col_str


def modify_JSON_file(json_file, sheet_index, color, cells):
    # Load the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    # Define the range for the update
    start_cell, end_cell = cells.split(':')
    start_column = re.findall(r'[A-Z]+', start_cell)[0]
    start_row = int(re.findall(r'\d+', start_cell)[0])
    end_column = re.findall(r'[A-Z]+', end_cell)[0]
    end_row = int(re.findall(r'\d+', end_cell)[0])

    # Convert column letters to numbers
    start_col_num = column_to_number(start_column)
    end_col_num = column_to_number(end_column)

    # Define the new colors
    new_start_color = color
    new_end_color = color

    # Get the sheet name (assuming the first one should be modified)
    sheet_name = list(data.keys())[sheet_index]

    # Iterate through the specified cells and update the fill color
    for row in range(start_row, end_row + 1):
        # Iterate through column numbers
        for col_num in range(start_col_num, end_col_num + 1):
            # Convert number back to Excel column letter
            cell_name = f"{number_to_column(col_num)}{row}"
            if cell_name in data[sheet_name]['cells']:  # Check if the cell exists
                data[sheet_name]['cells'][cell_name]['fill_color']['start_color'] = new_start_color
                data[sheet_name]['cells'][cell_name]['fill_color']['end_color'] = new_end_color

    # Save the updated data back to JSON
    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":

    # 常温模板
    # json_file = "template_details.json"
    # modify_JSON_file(json_file, 0, "FFd9d9d9", "A1:A23")    # 灰
    # modify_JSON_file(json_file, 0, "FFd9d9d9", "B1:S1")
    # modify_JSON_file(json_file, 0, "FFd9d9d9", "B9:Q9")
    # modify_JSON_file(json_file, 0, "FFd9d9d9", "B17:K17")
    # modify_JSON_file(json_file, 1, "FFd9d9d9", "F1:L1")
    # modify_JSON_file(json_file, 0, "FFbdd7ee", "B2:H8")     # 蓝
    # modify_JSON_file(json_file, 0, "FFf8cbad", "C10:Q16")   # 粉
    # modify_JSON_file(json_file, 0, "FFc6e0b4", "I2:R8")     # 绿
    # modify_JSON_file(json_file, 0, "FFc6e0b4", "B10:B16")
    # modify_JSON_file(json_file, 0, "FFffe699", "S3:S6")     # 黄
    # modify_JSON_file(json_file, 2, "FFffe699", "E10:K10")
    # modify_JSON_file(json_file, 2, "FFffe699", "E12:K12")
    # modify_JSON_file(json_file, 0, "FFbdd7ee", "K18:K18")
    # modify_JSON_file(json_file, 0, "FFf8cbad", "K19:K19")
    # modify_JSON_file(json_file, 0, "FFc6e0b4", "K20:K20")

    # 高低温模板
    json_file = "template_details_low_temp.json"
    modify_JSON_file(json_file, 0, "FFd9d9d9", "A1:AV1")    # 灰
    modify_JSON_file(json_file, 0, "FFd9d9d9", "A2:A52")
    modify_JSON_file(json_file, 0, "FFbdd7ee", "B2:B52")     # 蓝
    modify_JSON_file(json_file, 0, "FFbdd7ee", "D2:I52")
    modify_JSON_file(json_file, 0, "FFf8cbad", "J2:AD52")   # 粉
    modify_JSON_file(json_file, 0, "FFc6e0b4", "C2:C52")     # 绿
    modify_JSON_file(json_file, 0, "FFc6e0b4", "AE2:AM52")
    modify_JSON_file(json_file, 0, "FFffe699", "AN2:AN52")     # 黄
