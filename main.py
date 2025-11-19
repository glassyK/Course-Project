from datasets import load_dataset
import pandas as pd
import re


df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
ds_pr = load_dataset("hao-li/AIDev", "all_pull_request", split="train")
ds_repo = load_dataset("hao-li/AIDev", "all_repository", split="train")
ds_task = load_dataset("hao-li/AIDev", "pr_task_type", split="train")
ds_commit = load_dataset("hao-li/AIDev", "pr_commit_details", split="train")

# Convert to Pandas DataFrames for easier manipulation
df_pr = ds_pr.to_pandas()
df_repo = ds_repo.to_pandas()
df_task = ds_task.to_pandas()
df_commit = ds_commit.to_pandas()

df_task1 = df_pr.rename(columns={
    'title': 'TITLE', 'id': 'ID', 'agent': 'AGENTNAME',
    'body': 'BODYSTRING', 'repo_id': 'REPOID', 'repo_url': 'REPOURL'
})[['TITLE', 'ID', 'AGENTNAME', 'BODYSTRING', 'REPOID', 'REPOURL']]
df_task1.to_csv('task1_pull_requests.csv', index=False)

df_task2 = df_repo.rename(columns={
    'id': 'REPOID',
    'language': 'LANG',
    'stars': 'STARS',
    'url': 'REPOURL'
})[['REPOID', 'LANG', 'STARS', 'REPOURL']]
df_task2.to_csv('task2_repository.csv', index=False)

df_task3 = df_task.rename(columns={
    'id': 'PRID',
    'title': 'PRTITLE',
    'reason': 'PRREASON',
    'type': 'PRTYPE',
    'confidence': 'CONFIDENCE'
})[['PRID', 'PRTITLE', 'PRREASON', 'PRTYPE', 'CONFIDENCE']]
df_task3.to_csv('task3_pr_task_type.csv', index=False)

def clean_diff_patch(patch_string):
    if pd.isna(patch_string) or patch_string is None:
        return ""
    # Remove all non-ASCII characters and control characters (conservative cleaning)
    return re.sub(r'[^\x20-\x7E]+', ' ', str(patch_string))

df_commit['PRDIFF'] = df_commit['patch'].apply(clean_diff_patch)
df_task4 = df_commit.rename(columns={
    'pr_id': 'PRID', 'sha': 'PRSHA', 'message': 'PRCOMMITMESSAGE',
    'filename': 'PRFILE', 'status': 'PRSTATUS', 'additions': 'PRADDS',
    'deletions': 'PRDELSS', 'changes': 'PRCHANGECOUNT'
})[['PRID', 'PRSHA', 'PRCOMMITMESSAGE', 'PRFILE', 'PRSTATUS', 'PRADDS', 'PRDELSS', 'PRCHANGECOUNT', 'PRDIFF']]

df_task4.to_csv('task4_commit_details.csv', index=False)

SECURITY_KEYWORDS = [
    'race', 'racy', 'buffer', 'overflow', 'stack', 'integer', 'signedness',
    'underflow', 'improper', 'unauthenticated', 'gain access', 'permission',
    'cross site', 'css', 'xss', 'denial service', 'dos', 'crash', 'deadlock',
    'injection', 'request forgery', 'csrf', 'xsrf', 'forged', 'security',
    'vulnerability', 'vulnerable', 'exploit', 'attack', 'bypass', 'backdoor',
    'threat', 'expose', 'breach', 'violate', 'fatal', 'blacklist',
    'overrun', 'insecure'
]

df_merged = pd.merge(
    df_task1[['ID', 'AGENTNAME', 'TITLE', 'BODYSTRING']],
    df_task3[['PRID', 'PRTYPE', 'CONFIDENCE']],
    left_on='ID',
    right_on='PRID',
    how='inner'
)

def set_security_flag(row):
    text = str(row['TITLE']) + " " + str(row['BODYSTRING'])
    text = text.lower()
    for keyword in SECURITY_KEYWORDS:
        if keyword in text:
            return 1
    return 0

df_merged['SECURITY'] = df_merged.apply(set_security_flag, axis=1)

df_task5 = df_merged.rename(columns={
    'AGENTNAME': 'AGENT',
    'PRTYPE': 'TYPE',
})[['ID', 'AGENT', 'TYPE', 'CONFIDENCE', 'SECURITY']]

df_task5.to_csv('task5_security_flagged.csv', index=False)