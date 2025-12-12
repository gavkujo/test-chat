# data/parser_test.py

'''
Date Parser

This module provides utilities for extracting and normalizing settlement plate IDs and date inputs
from free-form text commands. It supports:

1. Plate ID extraction:
   - Uses a regex to match plates in the format F3-R##x-SM-##, where numbers and letters
     adhere to specific constraints.
   - Handles single and multi-plate scenarios.

2. Date normalization:
   - Converts various user-input date formats into ISO YYYY-MM-DD.
   - Supports natural language dates (e.g., "today", "tomorrow") and partial dates.
   - Infers missing years using reference dates and context-aware logic.
   - Validates chronological order of related date slots (SCD < ASD < max_date).

3. Function call generation:
   - Builds structured function calls (Asaoka_data, reporter_Asaoka, plot_combi_S, SM_Overview)
     from extracted plates and normalized dates.
   - Includes error handling for missing slots or conflicting classifications.
'''

import re
from dateparser import parse as date_parse

# Plate regex, with your constraints (1-80 for plate number)
PLATE_REGEX = r'F3\-R\d{2}[a-z]{1}\-SM\-(?:[0-9][0-9]?|80)'

examples = [
("this is prolly invalid but try anyway", "None"),
("I want a graph with the following plates: F3-R03a-SM-54. Only include data before July 22 2025.", "plot_combi_S"),
("Give me a snapshot of F3-R01d-SM-25 at a glance. From surcharge 24/06/24, ASD Apr 23, 2025, till max date 13/10/25.","Asaoka_data"),
("Generate a detailed PDF for the following plates: F3-R16b-SM-21, F3-R45c-SM-70, surcharge completed August 31, assessment from June 15 2025 until 16 Nov.","reporter_Asaoka"),
("Break down the Asaoka values for F3-R00b-SM-78 pls — dates are 02st March for surcharge, 16 May 2024 for assessment start, up to 17/08/24.","Asaoka_data"),
("yo lowkey thinkin bout settlements rn","None"),
("Hi system, initiate doc generation for F3-R22c-SM-77, F3-R37d-SM-23, F3-R36a-SM-22, F3-R10d-SM-35, F3-R38a-SM-41. Assessment phase Oct 19, 2024 - Aug 16, post surcharge 14 October.","reporter_Asaoka"),
("Give me a combined plot for plates F3-R31b-SM-54, F3-R40a-SM-63, F3-R44c-SM-57, assess until 03 February 2025.","plot_combi_S"),
("Provide a graph for settlement plates F3-R17c-SM-46 using data up to 10 Aug.","plot_combi_S"),
("Can I get the current stats for F3-R43b-SM-14? Looking for stuff like last settlement, GL, slope etc. Dates: SCD Jan 13, ASD Mar 22, 2025, till Sep 4, 2024.","Asaoka_data"),
("I wanna know everything measurable for F3-R21c-SM-58 — pairs, slope, r2, settlement. SCD=15st January, ASD=18 August 2024, max=02 Dec.","Asaoka_data"),
("literally no idea what this does","None"),
("Give me a snapshot of F3-R37b-SM-52 at a glance. From surcharge Oct 5, ASD 29 December, till max date Apr 5, 2024.","Asaoka_data"),
("Plot settlements for: F3-R15c-SM-33. Cutoff date: January 28 2024.","plot_combi_S"),
("science stuff please","None"),
("I wanna know everything measurable for F3-R07a-SM-01 — pairs, slope, r2, settlement. SCD=26/04/24, ASD=Feb 16, 2024, max=07st March.","Asaoka_data"),
("Multi-plate graph for: F3-R42b-SM-20, F3-R22a-SM-79, F3-R29a-SM-04, last data on 13/10/24.","plot_combi_S"),
("plot plates F3-R06c-SM-33 with max date Aug 27 thanks bby","plot_combi_S"),
("Do a single-plate assessment run on F3-R41d-SM-40 — show prediction, slope, intercept, whatever u got. Timeframe: 18 June–29-12-2024, SCD: July 10 2024.","Asaoka_data"),
("I wanna know everything measurable for F3-R32c-SM-37 — pairs, slope, r2, settlement. SCD=July 7 2025, ASD=August 7 2024, max=22 Feb.","Asaoka_data"),
("Graph time : plates F3-R01c-SM-39, F3-R07c-SM-32, stop on 08 Jul.","plot_combi_S"),
("What are the model results for F3-R00c-SM-77? I'm talkin m, b, r2, predicted value. Use dates 07 August 2025, August 19, 28 Aug.","Asaoka_data"),
("Provide a graph for settlement plates F3-R36d-SM-52, F3-R43a-SM-51, F3-R20a-SM-58, F3-R07b-SM-73 using data up to Jan 29, 2025.","plot_combi_S"),
("Need a doc for F3-R34a-SM-08, F3-R24d-SM-80, F3-R19d-SM-70 from 21/07/25 to 17-06-2024. Surcharge wrapped on 10th April.","reporter_Asaoka"),
("Plot this group: F3-R31d-SM-64, F3-R08b-SM-69, F3-R39c-SM-19. End data range on 30-07-2024.","plot_combi_S")
]

class MissingSlot(Exception):
    def __init__(self, slot):
        self.slot = slot

class FunctionClash(Exception):
    """Exception raised when there's a function classification clash"""
    def __init__(self, classifier_func, rule_func, original_input):
        self.classifier_func = classifier_func
        self.rule_func = rule_func
        self.original_input = original_input
        super().__init__(f"Function clash: {classifier_func} vs {rule_func}")

def find_plates(text):
    """Extract plate IDs as list."""
    return re.findall(PLATE_REGEX, text)

# a helper month→num map to keep things clean
MONTH_MAP = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'sept': '09', 'oct': '10', 'nov': '11', 'dec': '12'
}

from datetime import datetime, timedelta

def normalize_date_input(date_str, ref_date=None, slot_name=None, feedback_list=None):
    """Parse user date input and convert to YYYY-MM-DD, using ref_date for year inference if needed.
    Uses regex first, then dateparser fallback with context-aware logic.
    """
    date_str = date_str.strip()
    lower = date_str.lower()

    # Handle explicit "today", "tomorrow", etc.
    if lower in {"today", "now"}:
        dt = datetime.now()
        if feedback_list is not None:
            feedback_list.append(f"Inferred '{slot_name}' as today: {dt.date().isoformat()}")
        return dt.date().isoformat()
    if lower in {"tomorrow"}:
        dt = datetime.now() + timedelta(days=1)
        if feedback_list is not None:
            feedback_list.append(f"Inferred '{slot_name}' as tomorrow: {dt.date().isoformat()}")
        return dt.date().isoformat()

    date_patterns = [
        r"(\d{2})[\-/](\d{2})[\-/](\d{4})",        # DD-MM-YYYY or DD/MM/YYYY
        r"(\d{4})[\-/](\d{2})[\-/](\d{2})",        # YYYY-MM-DD or YYYY/MM/DD
        r"(\d{2})[\-/](\d{2})",                    # DD-MM or DD/MM (no year)
        r"(\d{1,2})\s+([A-Za-z]+)\s*(\d{4})?",     # 16 Aug OR 16 Aug 2024
        r"([A-Za-z]+)\s+(\d{2,4})",                # Aug 24 or August 2024
        r"([A-Za-z]+)\s+(\d{1,2})",                # Aug 16
        r"(\d{1,2})\s+([A-Za-z]+)"                 # 16 August
    ]

    # Try regex patterns first
    for pat in date_patterns:
        m = re.search(pat, date_str)
        if not m:
            continue

        # 1) DD-MM-YYYY
        if pat == date_patterns[0]:
            d, mth, y = m.groups()
            return f"{y}-{mth.zfill(2)}-{d.zfill(2)}"

        # 2) YYYY-MM-DD
        if pat == date_patterns[1]:
            y, mth, d = m.groups()
            return f"{y}-{mth.zfill(2)}-{d.zfill(2)}"

        # 3) DD-MM (no year → infer year)
        if pat == date_patterns[2]:
            d, mth = m.groups()
            base_year = 2025
            if ref_date:
                ref_year = date_parse(ref_date).year
                base_year = ref_year
            # Try to bump year if needed
            try:
                candidate = datetime(year=base_year, month=int(mth), day=int(d))
                if ref_date and candidate.date() < date_parse(ref_date).date():
                    candidate = candidate.replace(year=base_year + 1)
                return candidate.date().isoformat()
            except Exception:
                return None

        # 4) DD Month [Year]
        if pat == date_patterns[3]:
            d, month, y = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            year = y if y and len(y)==4 else None
            if not year:
                year = 2025
                if ref_date:
                    year = date_parse(ref_date).year
            try:
                candidate = datetime(year=int(year), month=int(mnum), day=int(d))
                if not y and ref_date and candidate.date() < date_parse(ref_date).date():
                    candidate = candidate.replace(year=int(year) + 1)
                return candidate.date().isoformat()
            except Exception:
                return None

        # 5) Month Year
        if pat == date_patterns[4]:
            month, year = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            y = year if len(year)==4 else f"20{year[-2:]}"
            try:
                candidate = datetime(year=int(y), month=int(mnum), day=1)
                return candidate.date().isoformat()
            except Exception:
                return None

        # 6) Month DD
        if pat == date_patterns[5]:
            month, d = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            base_year = 2025
            if ref_date:
                base_year = date_parse(ref_date).year
            try:
                candidate = datetime(year=base_year, month=int(mnum), day=int(d))
                if ref_date and candidate.date() < date_parse(ref_date).date():
                    candidate = candidate.replace(year=base_year + 1)
                return candidate.date().isoformat()
            except Exception:
                return None

        # 7) DD Month
        if pat == date_patterns[6]:
            d, month = m.groups()
            mon = month.lower()[:3]
            mnum = MONTH_MAP.get(mon, '01')
            base_year = 2025
            if ref_date:
                base_year = date_parse(ref_date).year
            try:
                candidate = datetime(year=base_year, month=int(mnum), day=int(d))
                if ref_date and candidate.date() < date_parse(ref_date).date():
                    candidate = candidate.replace(year=base_year + 1)
                return candidate.date().isoformat()
            except Exception:
                return None

    # fallback to dateparser for the wild ones (natural language, ambiguous, etc)
    settings = {'PREFER_DATES_FROM': 'future'}
    ref_dt = date_parse(ref_date) if ref_date else None
    if ref_dt:
        settings['RELATIVE_BASE'] = ref_dt
    else:
        settings['RELATIVE_BASE'] = datetime.now()
    dt = date_parse(date_str, settings=settings)
    if not dt:
        if feedback_list is not None:
            feedback_list.append(f"Could not parse '{slot_name}' from input: '{date_str}'")
        return None

    date_obj = dt.date()
    # If user gave a year, always use it
    explicit_year = re.search(r"\b(20\d{2})\b", date_str)
    if explicit_year:
        if feedback_list is not None:
            feedback_list.append(f"Inferred '{slot_name}' as {date_obj.isoformat()} (explicit year in input)")
        return date_obj.isoformat()

    # If no year, always parse with reference year, bump if needed
    if ref_dt:
        try:
            forced = date_obj.replace(year=ref_dt.year)
        except Exception:
            forced = date_obj.replace(month=3, day=1, year=ref_dt.year)
        if forced < ref_dt.date():
            try:
                forced = forced.replace(year=ref_dt.year + 1)
            except Exception:
                forced = forced.replace(month=3, day=1, year=ref_dt.year + 1)
        if feedback_list is not None:
            feedback_list.append(f"Inferred '{slot_name}' as {forced.isoformat()} (relative to {ref_dt.date().isoformat()})")
        return forced.isoformat()
    else:
        if feedback_list is not None:
            feedback_list.append(f"Inferred '{slot_name}' as {date_obj.isoformat()} (no reference date)")
        return date_obj.isoformat()

def validate_date_order(scd, asd, max_date, feedback_list=None):
    d1 = date_parse(scd).date()
    d2 = date_parse(asd).date()
    d3 = date_parse(max_date).date()
    # Friendly error if order is wrong
    if not (d1 < d2 < d3):
        order = sorted([('SCD', d1), ('ASD', d2), ('max_date', d3)], key=lambda x: x[1])
        msg = "Invalid date order. Did you mean: "
        msg += " < ".join(f"{name}={dt}" for name, dt in order)
        if feedback_list is not None:
            feedback_list.append(msg)
        raise ValueError(msg)
    # Warn if SCD and ASD are more than 40 days apart (usually within a month)
    if (d2 - d1).days > 40:
        warn = f"⚠️ Warning: SCD ({d1}) and ASD ({d2}) are more than 40 days apart. Usually these are within a month."
        if feedback_list is not None:
            feedback_list.append(warn)

def input_date_slot(slot_name):
    """Prompt user for date input and normalize it."""
    while True:
        raw = input(f"Enter {slot_name} (date): ")
        val = normalize_date_input(raw)
        if val:
            return val
        print("Invalid date format. Please try again.")

def build_function_call(func_name, plates, dates):
    """Generate the final python function call string."""
    if func_name == 'Asaoka_data':
        id_ = plates[0] if plates else None
        return f"Asaoka_data(id='{id_}', SCD='{dates['SCD']}', ASD='{dates['ASD']}', max_date='{dates['max_date']}')"
    
    elif func_name == 'reporter_Asaoka':
        ids_list = ", ".join(f"'{p}'" for p in plates)
        return f"reporter_Asaoka(ids=[{ids_list}], SCD='{dates['SCD']}', ASD='{dates['ASD']}', max_date='{dates['max_date']}')"
    
    elif func_name == 'plot_combi_S':
        ids_list = ", ".join(f"'{p}'" for p in plates)
        return f"plot_combi_S(ids=[{ids_list}], max_date='{dates['max_date']}')"
    
    elif func_name == 'SM_overview':
        ids_list = ", ".join(f"'{p}'" for p in plates)
        return f"SM_Overview(ids=[{ids_list}])"
    
    else:
        return f"# Unknown function: {func_name}"
    
def parse_and_build(user_text: str, func_name: str):
    # Split lines
    lines = user_text.splitlines()
    # The original user input is always the first line (or block before any slot-filling lines)
    orig_lines = []
    slot_lines = []
    for line in lines:
        if re.match(r'^(SCD|ASD|max_date):', line.strip(), re.IGNORECASE):
            slot_lines.append(line)
        else:
            orig_lines.append(line)
    orig_text = '\n'.join(orig_lines)

    plates = find_plates(orig_text)
    if not plates:
        raise MissingSlot('plates')

    needed_slots = {
        'Asaoka_data': ['SCD', 'ASD', 'max_date'],
        'reporter_Asaoka': ['SCD', 'ASD', 'max_date'],
        'plot_combi_S': ['max_date'],
        'SM_overview': []
    }

    if func_name not in needed_slots:
        raise ValueError(f"Function '{func_name}' not supported.")

    # Only check for slot values in the slot-filling context (never extract from original user input)
    slot_values = {}
    ref_dates = {}
    feedback = []
    for slot in needed_slots[func_name]:
        found = False
        for sline in reversed(slot_lines):
            m = re.match(rf'^{slot}:\s*(.+)$', sline.strip(), re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                # Normalize date for date slots, using SCD as reference for ASD/max_date
                if slot in ['SCD', 'ASD', 'max_date']:
                    if slot == 'SCD':
                        norm = normalize_date_input(val, slot_name='SCD', feedback_list=feedback)
                        ref_dates['SCD'] = norm
                    elif slot == 'ASD':
                        norm = normalize_date_input(val, ref_date=ref_dates.get('SCD'), slot_name='ASD', feedback_list=feedback)
                        ref_dates['ASD'] = norm
                    elif slot == 'max_date':
                        norm = normalize_date_input(val, ref_date=ref_dates.get('ASD') or ref_dates.get('SCD'), slot_name='max_date', feedback_list=feedback)
                    if not norm:
                        if feedback:
                            raise ValueError("\n".join(feedback))
                        raise MissingSlot(slot)  # Will re-prompt if invalid
                    slot_values[slot] = norm
                else:
                    slot_values[slot] = val
                found = True
                break
        if not found:
            raise MissingSlot(slot)

    # Sanity check date order if all present
    if func_name in ['Asaoka_data', 'reporter_Asaoka']:
        validate_date_order(slot_values['SCD'], slot_values['ASD'], slot_values['max_date'])
    if feedback:
        print("\n".join(feedback))


    # Return params dict with normalized date values
    if func_name == 'Asaoka_data':
        return {'id': plates[0], 'SCD': slot_values['SCD'], 'ASD': slot_values['ASD'], 'max_date': slot_values['max_date']}
    elif func_name == 'reporter_Asaoka':
        return {'ids': plates, 'SCD': slot_values['SCD'], 'ASD': slot_values['ASD'], 'max_date': slot_values['max_date']}
    elif func_name == 'plot_combi_S':
        return {'ids': plates, 'max_date': slot_values['max_date']}
    elif func_name == 'SM_overview':
        return {'ids': plates}
    else:
        raise ValueError(f"Function '{func_name}' not supported.")


def interactive_date_tester():
    print("=== Interactive Date Parser Tester ===")
    print("Enter a reference date (YYYY-MM-DD) or leave blank for today:")
    ref_date = input("Reference date: ").strip() or None
    while True:
        date_str = input("\nEnter a date string (or 'q' to quit): ").strip()
        if date_str.lower() in {"q", "quit", "exit"}:
            print("Exiting tester.")
            break
        feedback = []
        result = normalize_date_input(date_str, ref_date=ref_date, slot_name="test", feedback_list=feedback)
        print(f"Parsed: {result}")
        if feedback:
            print("Feedback:")
            for f in feedback:
                print("  -", f)

if __name__ == "__main__":
    interactive_date_tester()



