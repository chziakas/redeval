import replicate


class Mistral7bCompletion:
    def get_completion_from_messages(self, query):

        response_tokens = replicate.run(
            "mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70",
            input={"prompt": query},
        )
        combined_output = "".join(response_tokens)
        print(combined_output)
        return combined_output
