# -*- coding: utf-8 -*-
import json
import tensorflow as tf
import numpy as np

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        bounding_boxes_config = pb_utils.get_output_config_by_name(
            model_config, "bounding_boxes"
        )
        classes_names_config = pb_utils.get_output_config_by_name(
            model_config, "classes_names"
        )

        # Convert Triton types to numpy types
        self.bounding_boxes_dtype = pb_utils.triton_string_to_numpy(
            bounding_boxes_config["data_type"]
        )

        self.classes_names_dtype = pb_utils.triton_string_to_numpy(
            classes_names_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        threshold = float(self.model_config["parameters"]["THRESHOLD"]["string_value"])
        iouthreshold = float(
            self.model_config["parameters"]["IOUTHRESHOLD"]["string_value"]
        )
        label_map_path = self.model_config["parameters"]["LABEL_MAP_PATH"][
            "string_value"
        ]
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            num_detections = pb_utils.get_input_tensor_by_name(
                request, "num_detections"
            )
            detection_scores = pb_utils.get_input_tensor_by_name(
                request, "detection_scores"
            )
            detection_classes = pb_utils.get_input_tensor_by_name(
                request, "detection_classes"
            )
            detection_boxes = pb_utils.get_input_tensor_by_name(
                request, "detection_boxes"
            )
            
            input_tensor = pb_utils.get_input_tensor_by_name(
                request, "input_tensor"
            )

            input_img_width = input_tensor.shape()[1]
            input_img_height = input_tensor.shape()[2]
            
            detection_classes = detection_classes.as_numpy()
            detection_boxes = detection_boxes.as_numpy()
            num_detections = num_detections.as_numpy()
            detection_scores = detection_scores.as_numpy()
            length = len([i for i in detection_scores[0] if i > threshold])

            boxes_tensor = tf.convert_to_tensor(detection_boxes[0][:length], np.float32)
            scores_tensor = tf.convert_to_tensor(
                detection_scores[0][:length], np.float32
            )
            class_tensor = tf.convert_to_tensor(
                detection_classes[0][:length], np.float32
            )
            selected_indices, _ = tf.image.non_max_suppression_with_scores(
                boxes=boxes_tensor,
                max_output_size=length,
                score_threshold=threshold,
                iou_threshold=iouthreshold,
                scores=scores_tensor,
            )
            selected_boxes = tf.gather(boxes_tensor, selected_indices)
            selected_cls = tf.gather(class_tensor, selected_indices)
            np_boxes = selected_boxes.numpy()
            np_cls = selected_cls.numpy()

            boxes = np_boxes.tolist()
            abs_boxes = []
            for box in boxes:
                left, right, top, bottom = (
                    box[1] * input_img_width,
                    box[3] * input_img_width,
                    box[0] * input_img_height,
                    box[2] * input_img_height,
                )
                abs_boxes.append((left, right, top, bottom))

            # Create a reverse mapping (swap keys and values)
            reverse_mapping_dict = {v: k for k, v in label_map.items()}
            class_list = np_cls.tolist()

            # Map the provided value to its corresponding key using the reverse mapping
            mapped_label = [reverse_mapping_dict.get(value) for value in class_list]

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_bounding_boxes = pb_utils.Tensor(
                "bounding_boxes",
                np.array(abs_boxes).astype(self.bounding_boxes_dtype),
            )
            out_classes_names = pb_utils.Tensor(
                "classes_names", np.array(mapped_label).astype(self.classes_names_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_bounding_boxes, out_classes_names]
            )
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
