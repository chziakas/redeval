import replicate


class Llama270BCompletion:
    def get_completion_from_messages(self, query):

        response_tokens = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": query},
        )
        combined_output = "".join(response_tokens)
        print(combined_output)
        return combined_output
