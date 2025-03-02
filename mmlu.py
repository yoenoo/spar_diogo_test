from typing import Any, Union, Literal
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, Dataset, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, generate

def record_to_sample(record: dict[str, Any]) -> Sample:
  return Sample(
    input = record["question"],
    choices = record["choices"],
    target = ("ABCD"[record["answer"]]),
    metadata = {"subject": record["subject"]},
  )

def get_mmlu_dataset(
  split: Union[Literal["test"], Literal["dev"], Literal["validation"]] = "test",
  shuffle: bool = False,
  subjects: Union[list[str], str] = [],
) -> Dataset:
  dataset = hf_dataset(
    path="cais/mmlu",
    name="all",
    split=split,
    sample_fields=record_to_sample,
    shuffle=shuffle,
    seed=42,
  )

  subjects = subjects if isinstance(subjects, list) else [subjects]
  if len(subjects) > 0:
    return dataset.filter(
      name=f"{dataset.name}-{'-'.join(subjects)}",
      predicate=lambda s: s.metadata is not None and s.metadata.get("subject") in subjects
    )
  else:
    return dataset


@task 
def mmlu_0_shot(subjects: str | list[str] = [], cot: bool = False) -> Task:
  return Task(
    dataset=get_mmlu_dataset("test", shuffle=True, subjects=subjects),
    solver=multiple_choice(cot=cot),
    scorer=choice(),
    config=GenerateConfig(temperature=0.5),
  )

@task
def mmlu_5_shot(subjects: str | list[str] = []) -> Task:
  return Task(
    dataset=get_mmlu_dataset("test", shuffle=True, subjects=subjects),
    solver=[mmlu_5_shot_solver(), generate(max_tokens=1),]
    scorer=exact(),
    config=GenerateConfig(temperature=0),
  )

log = eval(mmlu_0_shot(), model="openai/gpt-4o-mini", limit=10)
print(log)