
from dreamsboard.dreams.task_step_md.base import TaskStepMD

from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore

from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import get_query_hash
import os

if __name__ == "__main__":
        
    start_task_context = os.environ.get("start_task_context")
    base_path = f'./{get_query_hash(start_task_context)}/'
    task_step_store = SimpleTaskStepStore.from_persist_dir(f'./{base_path}/storage')
    task_step_md = TaskStepMD(task_step_store)
    md_text =   task_step_md.format_md()
    print(md_text.text)
