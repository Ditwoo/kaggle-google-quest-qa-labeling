from catalyst.dl import SupervisedRunner


class Runner(SupervisedRunner):
    def __init__(self, model=None, device=None):
        super().__init__(
            model=model, 
            device=device, 
            # input_key=["question_title", "question_body", "answer"], 
            # input_key=["question_title", "question_body", "answer", "category", "host"], 
            # output_key="logits",
        )