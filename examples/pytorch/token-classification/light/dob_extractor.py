import time
import datetime
from collections import defaultdict
import dateparser
from functools import lru_cache


class DOBExtractor():
    DATE_RELATED_WORDS = {"th", "rd", "st", "of"}
    
    @lru_cache(maxsize=1024)
    def cached_parse_date(self, string):
        return dateparser.parse(string)

    def is_symmetry_str(self, target_str):
        words = target_str.split(" ")
        return words == words[::-1]

    def reform_the_info(self, info_list: list):
        reformed_info = defaultdict(list)
        for result_item in info_list:
            reformed_info[result_item["line_index"]].extend(result_item["word_indexes"])
        return reformed_info

    def deep_filter(self, word_indexes, line_words):
        word_indexes_list = []
        
        if "(" in line_words and ")" in line_words:
            return [word_indexes]
        
        alpha_cnt, digit_cnt, four_digit_cnt = 0, 0, 0
        
        for word_index in word_indexes:
            word = line_words[word_index]
            if word.isalpha() and word.lower() not in self.DATE_RELATED_WORDS:
                alpha_cnt += 1
            if word.isdigit():
                digit_cnt += 1
                if len(word) == 4:
                    four_digit_cnt += 1
        
        middle_index = len(word_indexes) // 2
        if alpha_cnt >= 2 and len(word_indexes) % 2 == 1 and self.cached_parse_date(line_words[middle_index]):
            word_indexes_list.append(word_indexes[:middle_index])
            word_indexes_list.append(word_indexes[middle_index + 1:])
        elif digit_cnt == 2 and four_digit_cnt == 2 and len(word_indexes) in {2, 3}:
            word_indexes_list.extend([[word_index] for word_index in word_indexes if line_words[word_index].isdigit()])
        elif (digit_cnt == 4 or digit_cnt == 6) and self.is_symmetry_str(" ".join(line_words)):
            half = len(word_indexes) // 2
            word_indexes_list.append(word_indexes[:half])
            word_indexes_list.append(word_indexes[-half:])
        else:
            word_indexes_list.append(word_indexes)
        
        return word_indexes_list

    def filter_words(self, word_indexes, line_words):
        while word_indexes:
            if line_words[word_indexes[0]].isalpha() and not self.cached_parse_date(line_words[word_indexes[0]]):
                word_indexes.pop(0)
            elif line_words[word_indexes[-1]].isalpha() and not self.cached_parse_date(line_words[word_indexes[-1]]):
                word_indexes.pop()
            else:
                return self.deep_filter(word_indexes, line_words)
        return []

    def filter_info(self, result_info, msgs):
        line_infos = self.reform_the_info(result_info)
        filtered_infos = []
        
        for line_index, word_indexes in line_infos.items():
            merged_line = sorted(set(word_indexes))
            pieces = []
            word_piece = []
            for word_index in merged_line:
                if not word_piece or word_index == word_piece[-1] + 1:
                    word_piece.append(word_index)
                else:
                    pieces.append(word_piece)
                    word_piece = [word_index]
            if word_piece:
                pieces.append(word_piece)

            line_words = msgs[line_index][1].split(" ")
            for word_indexes in pieces:
                for filtered_word_indexes in self.filter_words(word_indexes.copy(), line_words):
                    words = " ".join(line_words[index] for index in filtered_word_indexes)
                    filtered_infos.append({
                        "words": words,
                        "line_index": line_index,
                        "word_indexes": filtered_word_indexes,
                    })

        return filtered_infos

    def is_empty_dict(self, check_dict):   
        return all(value is None for value in check_dict.values())

    def extract_dob(self, age_list, return_unextracted=False):
        dob = {"year": None, "month": None, "day": None}
        age = None
        unextract_dict = {}
        age_item_final = None
        now = datetime.datetime.now()  # Call only once to get current date
        
        for item_index, age_str in enumerate(age_list):
            dob_dict_item, age_item = self._extract_single_age_str(age_str)
            unextracted = self.is_empty_dict(dob_dict_item) and age_item is None
            if unextracted:
                unextract_dict[item_index] = age_str
            if age_item is not None:
                age_item_final = age_item
            
            dob.update((k, v) for k, v in dob_dict_item.items() if v is not None)
            complete_dob = all(value is not None for value in dob.values())
            if item_index != 0 and complete_dob:
                if age is None:
                    age = now.year - dob["year"] - (1 if now.month < dob["month"] else 0)

                res = f"{dob['year']}-{dob['month']}-{dob['day']}"
                return res, age

        if age is None and dob["year"] is not None:
            age = now.year - dob["year"] - (1 if dob["month"] and now.month < dob["month"] else 0)
        if self.is_empty_dict(dob) and age is None:
            age = age_item_final
        res = f"{dob['year']}-{dob['month']}-{dob['day']}"
        return res, age

    def _extract_single_age_str(self, age_str):
        dob_dict = {"year": None, "month": None, "day": None}
        age = None
        
        if age_str.isdigit():
            age_int = int(age_str)
            if age_int < 100:
                if age_int < 10:
                    dob_dict["year"] = 2000 + age_int
                elif age_int <= 50:
                    age = age_int
                else:
                    dob_dict["year"] = 1900 + age_int
            elif 1920 < age_int < 2010:
                dob_dict["year"] = age_int
            else:
                # Assume the last two digits represent the year
                dob_dict["year"] = 1900 + (age_int % 100)
        else:
            date = self.cached_parse_date(age_str)
            if date and 1920 < date.year < 2010:
                dob_dict.update({"year": date.year, "month": date.month})
                if str(date.day) in age_str:
                    dob_dict["day"] = date.day

        if not any(dob_dict.values()):  # No value extracted yet, try illegal str extraction
            dob_dict, age = self._extract_illegal_str(age_str)

        return dob_dict, age

    def _extract_illegal_str(self, age_str):
        dob_dict = {"year": None, "month": None, "day": None}
        age = None

        # No changes needed if the string is not purely numeric and not the expected length
        if not age_str.isdigit() or len(age_str) not in (6, 8):
            return dob_dict, age

        # Extract potential year, month, and day substrings
        if len(age_str) == 8:
            possible_year = age_str[:4] if int(age_str[-4:]) > 1930 else age_str[-4:]
            dob_dict['year'] = int(possible_year)
            dob_dict['month'] = int(age_str[4:6] if int(age_str[-4:]) > 1930 else age_str[2:4])
            dob_dict['day'] = int(age_str[6:] if int(age_str[-4:]) > 1930 else age_str[:2])
        else:  # len(age_str) == 6
            dob_dict['year'] = int(age_str[:2])
            dob_dict['month'] = int(age_str[2:4])
            dob_dict['day'] = int(age_str[4:6])

        # Verify the extracted date
        extracted_date = f"{dob_dict['year']:04d}-{dob_dict['month']:02d}-{dob_dict['day']:02d}"
        parsed_date = self.cached_date_parse(extracted_date)

        # Verify if parsed_date falls within a reasonable range and matches the day if applicable
        if parsed_date and 1920 < parsed_date.year < 2010:
            dob_dict['year'] = parsed_date.year
            dob_dict['month'] = parsed_date.month
            dob_dict['day'] = parsed_date.day if str(parsed_date.day) in age_str else None

        return dob_dict, age
    

def test(samples):
    samples = samples
    extractor = DOBExtractor()
    start_time = time.time()
    for sample in samples:
        res = extractor.extract_dob([sample])
        print(f"input dobs: {sample}, output results: {res}")
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / len(samples)
    print(f"total cost time: {total_time}, total samples: {len(samples)}, average time: {average_time}")


if __name__ == "__main__":
    test_samples = [
        "2000 10 . 7",
        "February 11, 2004",
        "9-9-93",
        "3-15-84",
        "9 / 26 / 92",
        "oct 30 94",
        "05 / 09 / 1990",
        "'28 / 09 / 1998'"
        "22/01/89",
        "July 27th",
        "January 29th",
        "06/21/2004",
        "10/10/2003",
        "04/16/1999",
        '9 - 13 - 2004',
        "august 23 1998",
        'may 16th 2007',
        '08 / 10 / 01',
        '6 / 6 / 02',
        "Oct 5 th 1998",
        "82278",
        "3577",
        '1st of october 2003',
        '21 . 12 . 1999',
        'march 18 1998',
        'july 71995',
        "22 10 1982",
        "2 21974",
        "12-15-2001",
        "2nd April 1985",
        "June 23, 1958",
        "September 6, 1975",
        "15/10/78",
        "5th September 1970",
        "012694",
        "030396",
        "3-4-99",
        "August 20 2000",
        "9/19/93",
        "1979",
        "1980",
        "41",
        "46",
        "14/06/81",
        "20 May 1983",
        "4 June 1986",
        "9th of may",
        "28th of August",
        "august 26 1991",
        "September 27th 2002",
        "April 29",
    ]
    test(test_samples)