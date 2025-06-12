import pandas as pd
import json

def clean_excel_and_generate_json(file_path):
    df = pd.read_excel(file_path, dtype=str)

    df.columns = [col.strip() for col in df.columns]

    if 'Item' in df.columns:
        df['Item'] = df['Item'].ffill()
    else:
        raise KeyError("'Item' column not found")

    print("Top 5 rows:")
    print(df.head())

    expected_columns = ['Field', 'Size', 'Position']
    for col in expected_columns:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    original_len = len(df)
    df_clean = df.dropna(subset=expected_columns)
    dropped_rows = original_len - len(df_clean)
    if dropped_rows > 0:
        print(f"\nDropped {dropped_rows} rows due to missing values in {expected_columns}\n")

    df_clean.reset_index(drop=True, inplace=True)

    detail = {}
    for idx, row in df_clean.iterrows():
        try:
            field_name = row['Field'].strip().replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            size = int(row['Size'])
            pos = row['Position'].strip()

            detail[str(idx + 1)] = {
                "fieldName": field_name,
                "dataType": "string",
                "size": size,
                "pos": pos
            }
        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    return {"detail": detail}

excel_file_path = r"C:\Users\MANID\Documents\Type_D.xlsx"
output_json = clean_excel_and_generate_json(excel_file_path)

print(json.dumps(output_json, indent=4))

with open('output.json', 'w') as f:
    json.dump(output_json, f, indent=4)

