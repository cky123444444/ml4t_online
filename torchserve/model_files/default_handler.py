import torch
from ts.torch_handler.base_handler import BaseHandler

class DefaultHandler(BaseHandler):
    """
    TorchServe handler for tabular feature input.
    Expects input as a list or numpy array of shape [batch_size, input_dim].
    """

    def preprocess(self, data):
        # data: list of dicts, each with 'body' key containing input features
        input_data = []
        for row in data:
            # Accept both bytes and list
            body = row.get("body")
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")
            if isinstance(body, str):
                import json
                body = json.loads(body)
            input_data.append(body)
        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).squeeze(1)  # [batch_size, 1, input_dim]
        print(f"[DefaultHandler] Preprocessed input tensor shape: {input_tensor.shape}") # [1, 1, 111]
        return input_tensor

    def inference(self, model_input):
        # model_input: torch.FloatTensor [batch_size, input_dim]
        with torch.no_grad():
            outputs = self.model(model_input)
        return outputs

    # def postprocess(self, inference_output):
    #     # inference_output: tuple of tensors, each shape [batch, ...]
    #     keys = [
    #         "ctrl_return_1h_out",
    #         "trmt_return_1h_out",
    #         "ctrl_sigma_1h_out",
    #         "trmt_sigma_1h_out",
    #         "t_out"   
    #     ]
    #     sigmoid_keys = {"t_out"}
    #     # 假设 batch size = N
    #     batch_size = inference_output[0].shape[0]
    #     results = []
    #     for i in range(batch_size):
    #         result = {}
    #         for k, v in zip(keys, inference_output):
    #             val = v[i]
    #             if k in sigmoid_keys:
    #                 val = torch.sigmoid(val)
    #             result[k] = val.cpu().numpy().tolist()
    #         results.append(result)
    #     return results

    def postprocess(self, inference_output):
        # inference_output: tuple of tensors, each shape [batch, ...]
        keys = [
            "ctrl_return_1h_out",
            "trmt_return_1h_out",
            "ctrl_sigma_1h_out",
            "trmt_sigma_1h_out",
            "t_ctrl_out",
            "t_trmt_out" 
        ]
        sigmoid_keys = {"t_ctrl_out", "t_trmt_out"}
        # 假设 batch size = N
        batch_size = inference_output[0].shape[0]
        results = []
        for i in range(batch_size):
            result = {}
            for k, v in zip(keys, inference_output):
                val = v[i]
                if k in sigmoid_keys:
                    val = torch.sigmoid(val)
                result[k] = val.cpu().numpy().tolist()
            results.append(result)
        return results