import re
from datetime import datetime


def determine_data_type(column_data):
    data_types = {
        "Boolean": 0,
        "Number": 0,
        "Date": 0,
        "Time": 0,
        "DateTime": 0,
        "Text": 0
    }

    boolean_values = {"true", "false", "yes", "no", "1", "0"}
    number_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    date_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y.%m.%d"
    ]
    time_formats = [
        "%H:%M",
        "%I:%M %p",
        "%H:%M:%S"
    ]
    datetime_formats = [
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M %p"
    ]

    for item in column_data:
        item_str = str(item).strip().lower()

        # Check for Boolean
        if item_str in boolean_values:
            data_types["Boolean"] += 1
            continue

        # Check for Number
        if number_pattern.match(item_str):
            data_types["Number"] += 1
            continue

        # Check for DateTime
        for fmt in datetime_formats:
            try:
                datetime.strptime(item_str, fmt)
                data_types["DateTime"] += 1
                break
            except ValueError:
                continue
        else:
            # Check for Date
            for fmt in date_formats:
                try:
                    datetime.strptime(item_str, fmt)
                    data_types["Date"] += 1
                    break
                except ValueError:
                    continue
            else:
                # Check for Time
                for fmt in time_formats:
                    try:
                        datetime.strptime(item_str, fmt)
                        data_types["Time"] += 1
                        break
                    except ValueError:
                        continue
                else:
                    # Default to Text
                    data_types["Text"] += 1

    # Determine the most frequent data type
    determined_type = max(data_types, key=data_types.get)
    return determined_type
