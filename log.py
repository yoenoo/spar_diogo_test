from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_sample

logs = list_eval_logs()
for log in logs:
  print(log)
  
  log = read_eval_log_sample(log, id=1)
  break

print(log)