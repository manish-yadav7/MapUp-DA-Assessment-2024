from typing import Dict, List

import pandas as pd

import re

import math

import polyline


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    i = 0
    while i < len(lst):
        start = i
        end = min(i + n - 1, len(lst) - 1)
        while start < end:
            lst[start], lst[end] = lst[end], lst[start]
            start += 1
            end -= 1
        
        i += n
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    dict = {}  
    
    for string in lst:
        length = len(string)  
        
        if length not in dict:
            dict[length] = []
        
        dict[length].append(string)
    return dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(current, parent_key=''):
        items = {}
        for k, v in current.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.update(_flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.update(_flatten(item, f"{new_key}[{i}]"))
            else:
                items[new_key] = v
        return items

    flattened = _flatten(nested_dict)
    
    print("Flattened dictionary:", flattened)
    
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                
                nums[start], nums[i] = nums[i], nums[start]
                
                backtrack(start + 1)
                
                nums[start], nums[i] = nums[i], nums[start]
    
    nums.sort()
    result = []
    backtrack(0)
    
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',
        r'\b\d{2}/\d{2}/\d{4}\b',
        r'\b\d{4}\.\d{2}\.\d{2}\b'
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    return matches

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    def haversine_distance(lat1, lon1, lat2, lon2):

        R = 6371000  # Radius of Earth in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
    
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0]
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    df['distance'] = distances
    
    print(df)
    
    return pd.DataFrame()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    def rotate_matrix_90_clockwise(matrix: List[List[int]]) -> List[List[int]]:
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            matrix[i].reverse()
        return matrix
    
    def replace_with_sum(matrix: List[List[int]]) -> List[List[int]]:
        n = len(matrix)
        transformed_matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                row_sum = sum(matrix[i]) - matrix[i][j]
                col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]
                transformed_matrix[i][j] = row_sum + col_sum
        
        return transformed_matrix
    
    rotated_matrix = rotate_matrix_90_clockwise(matrix)
    
    final_matrix = replace_with_sum(rotated_matrix)
    
    return final_matrix

    return []


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df = pd.read_csv('datasets/dataset-1.csv')
    
    df_relevant = df[['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime']].copy()

    day_to_weekday = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    df_relevant['startDay'] = df_relevant['startDay'].map(day_to_weekday)
    df_relevant['endDay'] = df_relevant['endDay'].map(day_to_weekday)

    df_relevant['start_datetime'] = pd.to_datetime(df_relevant['startDay'].astype(str) + ' ' + df_relevant['startTime'], errors='coerce', format='%w %H:%M:%S')
    df_relevant['end_datetime'] = pd.to_datetime(df_relevant['endDay'].astype(str) + ' ' + df_relevant['endTime'], errors='coerce', format='%w %H:%M:%S')

    def check_coverage(group):
        covered_days = set()
        
        full_day_start = pd.Timestamp('00:00:00').time()
        full_day_end = pd.Timestamp('23:59:59').time()

        for _, row in group.iterrows():
            start_day = row['start_datetime'].weekday()
            covered_days.add(start_day)
            
            if row['start_datetime'].time() > full_day_start or row['end_datetime'].time() < full_day_end:
                return True

        if len(covered_days) < 7:
            return True
        
        return False

    result = df_relevant.groupby(['id', 'id_2']).apply(check_coverage)

    return result
