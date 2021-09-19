from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pandas as pd
import torch

df = pd.read_csv('athletes_cleaned.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
model = model.to(device)

def age_span(age):
    if int(age) < 20:
        return '10-19'
    elif int(age) < 30:
        return '20-29'
    elif int(age) < 40:
        return '30-39'
    else:
        return '40-'
df['age_span'] = df['age'].apply(lambda x:age_span(x))

countries = sorted(list(set(df['country'].tolist())))
sports = sorted(list(set(df['sports'].tolist())))

def conditional_summary(col_name):
    result_df = pd.DataFrame(columns=['condition', 'summary'])
    count = 0

    text = ' '.join([d for d in df[col_name].tolist() if d != 'none'])
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)

    # 전체 운동선수
    summary_ids = model.generate(**inputs)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    result_df.loc[count] = ['전체', summary[0]]
    count += 1

    # 연령대별 요약 
    ages = ['10-19', '20-29', '30-39', '40-']
    for age in ages:
        text = ' '.join([d for d in df.query('age_span==@age')[col_name].tolist() if d != 'none'])
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        summary_ids = model.generate(**inputs)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        result_df.loc[count] = [age, summary[0]]
        count += 1

    # 성별별 요약 
    genders = ['Female', 'Male']
    for gender in genders:
        text = ' '.join([d for d in df.query('gender==@gender')[col_name].tolist() if d != 'none'])
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        summary_ids = model.generate(**inputs)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        result_df.loc[count] = [gender, summary[0]]
        count += 1

    # 국적별 요약
    for country in countries:
        text = ' '.join([d for d in df.query('country==@country')[col_name].tolist() if d != 'none'])
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        summary_ids = model.generate(**inputs)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        result_df.loc[count] = [country, summary[0]]
        count += 1

    # 종목별 요약
    for sport in sports:
        text = ' '.join([d for d in df.query('sports==@sport')[col_name].tolist() if d != 'none'])
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        summary_ids = model.generate(**inputs)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        result_df.loc[count] = [sport, summary[0]]
        count += 1

    return result_df


conditional_summary('philosophy').to_csv('philosophy_summary.csv', index=False, encoding='utf-8')
conditional_summary('ambition').to_csv('ambition_summary.csv', index=False, encoding='utf-8')

