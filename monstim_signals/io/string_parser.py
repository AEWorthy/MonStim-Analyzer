import logging
import re
from datetime import datetime


def parse_date(date_string: str, preferred_format: str = None):
    """
    Parses a date string and returns a datetime object and its format.
    Args:
        date_string (str): The date string to parse. Must be 6 or 8 characters long.
        preferred_format (str, optional): The preferred format to return. Options are 'YYMMDD', 'DDMMYY', 'MMDDYY', 'YYYYMMDD', 'DDMMYYYY', 'MMDDYYYY'.
    Returns:
        tuple: A tuple containing the parsed datetime object and its format name, or None if the date string is invalid.
    Raises:
        ValueError: If the date string is not 6 or 8 characters long.
    """

    def is_valid_date(year, month, day):
        try:
            datetime(year, month, day)
            return True
        except ValueError:
            return False

    if len(date_string) == 6:
        formats = [("%y%m%d", "YYMMDD"), ("%d%m%y", "DDMMYY"), ("%m%d%y", "MMDDYY")]
    elif len(date_string) == 8:
        formats = [
            ("%Y%m%d", "YYYYMMDD"),
            ("%d%m%Y", "DDMMYYYY"),
            ("%m%d%Y", "MMDDYYYY"),
        ]
    else:
        return (
            None,
            f"Invalid date string length ('{date_string}'): must be 6 or 8 characters.",
        )

    valid_formats = []
    for date_format, format_name in formats:
        try:
            parsed_date = datetime.strptime(date_string, date_format)
            if is_valid_date(parsed_date.year, parsed_date.month, parsed_date.day):
                valid_formats.append((parsed_date, format_name))
        except ValueError:
            continue

    if not valid_formats:
        return (
            None,
            f"No valid date format found: please check the date string ('{date_string}').",
        )

    if len(valid_formats) == 1:
        return valid_formats[0]

    if preferred_format:
        for parsed_date, format_name in valid_formats:
            if format_name == preferred_format:
                return parsed_date, format_name

    # If we reach here, we have multiple valid formats and no preferred format
    logging.warning(
        f"Ambiguous date. Please specify preferred format: {[f for _, f in valid_formats]}. Returning first valid format: {valid_formats[0][1]}."
    )
    return valid_formats[0]


def parse_dataset_name(dataset_name: str, preferred_date_format: str = None) -> tuple:
    """
    Extracts information from a dataset' directory name.

    Args:
        dataset_name (str): The name of the dataset in the format '[YYMMDD] [AnimalID] [Condition]'.
        preferred_date_format (str, optional): The preferred date format to return. Options are 'YYMMDD', 'DDMMYY', 'MMDDYY', 'YYYYMMDD', 'DDMMYYYY', 'MMDDYYYY'.

    Returns:
        tuple: A tuple containing the extracted information in the following order: (date, animal_id, condition).
            If the dataset name does not match the expected format, returns (None, None, None).
    """
    # Define the regex pattern
    # The dataset name is expected to follow the format: '[YYMMDD] [AnimalID] [Condition]'
    # - (\d{6,8}): Captures the date, which can be 6 digits (YYMMDD) or 8 digits (YYYYMMDD).
    # - ([A-Z0-9.]+): Captures the animal ID, which consists of uppercase letters, digits, and periods.
    # - (.+): Captures the condition, which can be any sequence of characters.
    pattern = r"^(\d{6,8})\s([A-Z0-9.]+)\s(.+)$"

    # Match the pattern
    match = re.match(pattern, dataset_name)

    if match:
        date_string = match.group(1)
        animal_id = match.group(2)
        condition = match.group(3)

        parsed_date, format_info = parse_date(date_string, preferred_date_format)

        if isinstance(parsed_date, datetime):
            formatted_date = parsed_date.strftime("%Y-%m-%d")
            logging.debug(f"Date: {formatted_date} ({format_info}), Animal ID: {animal_id}, Condition: {condition}")
            return formatted_date, animal_id, condition
        else:
            logging.error(
                f"Error: {format_info}\n\n Make the neccessary changes to the dataset name, re-import your data, and try again."
            )
            # return 'DATE_ERROR', animal_id, condition
            raise ValueError(f"Error: {format_info}")

    else:
        logging.error(
            f"Error: Dataset ID '{dataset_name}' does not match the expected format: '[YYMMDD] [AnimalID] [Condition]'."
        )
        raise ValueError(
            f"Error: Dataset ID '{dataset_name}' does not match the expected format: '[YYMMDD] [AnimalID] [Condition]'."
        )
