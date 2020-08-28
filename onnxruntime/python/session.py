# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi import _pybind_state as C


def get_ort_device_type(device):
    if device == 'cuda':
        return C.OrtDevice.cuda()
    elif device == 'cpu':
        return C.OrtDevice.cpu()
    else:
        raise Exception('Unsupported device type: ' + device)


class Session:
    """
    This is the main class used to run a model.
    """
    def __init__(self, sess):
        self._sess = sess
        self._enable_fallback = True

    # TODO: Can we remove this? If we separate session creation from model loading/initialization
    # we shouldn't need any 'reset' functionality as the user can change provider info etc. before model load.
    def _reset_session(self):
        "release underlying session object."
        # meta data references session internal structures
        # so they must be set to None to decrement _sess reference count.
        self._inputs_meta = None
        self._outputs_meta = None
        self._overridable_initializers = None
        self._model_meta = None
        self._providers = None
        self._sess = None

        # At this point, _sess object is still referenced by _sess_options,
        # because of previously _sess_options = _sess.sess_options being executed in _load_model().
        # Therefore, _sess reference count is not zero and not being released by python gc yet.
        #
        # In order to make _sess reference count become 0 and being destroyed by python gc before
        # creating new session object, we need to reset _sess_options as well.
        self._sess_options = None
        self._sess_options = self._sess_options_initial

    def get_session_options(self):
        "Return the session options. See :class:`onnxruntime.SessionOptions`."
        return self._sess_options

    def get_inputs(self):
        "Return the inputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._inputs_meta

    def get_outputs(self):
        "Return the outputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._outputs_meta

    def get_overridable_initializers(self):
        "Return the inputs (including initializers) metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._overridable_initializers

    def get_modelmeta(self):
        "Return the metadata. See :class:`onnxruntime.ModelMetadata`."
        return self._model_meta

    def get_providers(self):
        "Return list of registered execution providers."
        return self._providers

    def get_provider_options(self):
        "Return registered execution providers' configurations."
        return self._provider_options

    # TODO: This may not be the best approach vs. simply requiring the providers and options to be specified
    # at construction time. Otherwise we're keeping memory unnecessarily (path_or_bytes could be bytes) and
    # causing unexpected side effects (recreating the underlying instance when setting an option)
    def set_providers(self, providers, provider_options=None):
        """
        Register the input list of execution providers. The underlying session is re-created.

        :param providers: list of execution providers
        :param provider_options: list of provider options dict for each provider, in the same order as 'providers'

        The list of providers is ordered by Priority. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """
        if not set(providers).issubset(C.get_available_providers()):
            raise ValueError("{} does not contain a subset of available providers {}".format(
                providers, C.get_available_providers()))

        if provider_options:
            if not isinstance(providers, list) or not isinstance(provider_options, list):
                raise ValueError("Inputs must be two python lists.")

            if len(providers) != len(provider_options):
                raise ValueError("Two input lists must have same length.")

            for option in provider_options:
                if not isinstance(option, dict):
                    raise ValueError("Provider options must be list of python dict.")

                for key, val in option.items():
                    option[key] = str(val)

        self._reset_session()
        self._load_model(providers, provider_options)

    def disable_fallback(self):
        """
        Disable session.run() fallback mechanism.
        """
        self._enable_fallback = False

    def enable_fallback(self):
        """
        Enable session.Run() fallback mechanism. If session.Run() fails due to an internal Execution Provider failure,
        reset the Execution Providers enabled for this session.
        If GPU is enabled, fall back to CUDAExecutionProvider.
        otherwise fall back to CPUExecutionProvider.
        """
        self._enable_fallback = True

    def run(self, output_names, input_feed, run_options=None):
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: See :class:`onnxruntime.RunOptions`.

        ::

            sess.run([output_name], {input_name: x})
        """
        num_required_inputs = len(self._inputs_meta)
        num_inputs = len(input_feed)
        # the graph may have optional inputs used to override initializers. allow for that.
        if num_inputs < num_required_inputs:
            raise ValueError("Model requires {} inputs. Input Feed contains {}".format(num_required_inputs, num_inputs))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return self._sess.run(output_names, input_feed, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print("EP Error: {} using {}".format(str(err), self._providers))
                print("Falling back to {} and retrying.".format(self._fallback_providers))
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                return self._sess.run(output_names, input_feed, run_options)
            else:
                raise

    def end_profiling(self):
        """
        End profiling and return results in a file.

        The results are stored in a filename if the option
        :meth:`onnxruntime.SessionOptions.enable_profiling`.
        """
        return self._sess.end_profiling()

    def io_binding(self):
        "Return an onnxruntime.IOBinding object`."
        return IOBinding(self)

    def run_with_iobinding(self, iobinding, run_options=None):
        """
         Compute the predictions.

         :param iobinding: the iobinding object that has graph inputs/outputs bind.
         :param run_options: See :class:`onnxruntime.RunOptions`.
        """
        self._sess.run_with_iobinding(iobinding._iobinding, run_options)


class InferenceSession(Session):
    """
    This is the main class used to run a model.
    """
    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None,
                 ort_format_model=False):
        """
        :param path_or_bytes: filename or serialized model in a byte string
        :param sess_options: session options
        :param providers: list of providers to use for session. If empty, will use all available providers.
        :param provider_options: list of provider options dict for each provider, in the same order as 'providers'
        :param ort_format_model: True if creating a session with an ORT format model. False for an ONNX format model.

        """

        if isinstance(path_or_bytes, str):
            self._model_path = path_or_bytes
            self._model_bytes = None
        elif isinstance(path_or_bytes, bytes):
            self._model_path = None
            self._model_bytes = path_or_bytes  # TODO: This is bad as we're holding the memory indefinitely
        else:
            raise TypeError("Unable to load from type '{0}'".format(type(path_or_bytes)))

        if ort_format_model:
            if (providers and len(providers) > 0) or (provider_options and len(provider_options) > 0):
                raise ValueError("ORT format model does not support specifying 'providers' or 'provider_options'")

        self._sess_options = sess_options
        self._sess_options_initial = sess_options
        self._enable_fallback = True
        self._ort_format_model = ort_format_model

        sess = C.InferenceSession(self._sess_options if self._sess_options else C.get_default_session_options())
        Session.__init__(self, sess)

        self._load_model(providers, provider_options)

    # TODO: Rethink this setup. If someone wants to create a session, change providers/provider options, and then
    # load the model we should facilitate that instead of always loading the model in the init and re-creating the
    # inference session when the providers/provider options change. This setup also means we have to hold a reference
    # to path_or_bytes forever which is not optimal if that is in-memory bytes.
    def _load_model(self, providers, provider_options):

        if self._model_path:
            if self._ort_format_model:
                self._sess.load_ort_model(self._model_path, True)
            else:
                self._sess.load_model(self._model_path, True, providers or [], provider_options or [])
        else:
            if self._ort_format_model:
                self._sess.load_ort_model(self._model_bytes, False)
            else:
                self._sess.load_model(self._model_bytes, False, providers or [], provider_options or [])

        self._sess_options = self._sess.session_options
        self._inputs_meta = self._sess.inputs_meta
        self._outputs_meta = self._sess.outputs_meta
        self._overridable_initializers = self._sess.overridable_initializers
        self._model_meta = self._sess.model_meta
        self._providers = self._sess.get_providers()
        self._provider_options = self._sess.get_provider_options()

        # Tensorrt can fall back to CUDA. All others fall back to CPU.
        if 'TensorrtExecutionProvider' in C.get_available_providers():
            self._fallback_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self._fallback_providers = ['CPUExecutionProvider']


class IOBinding:
    '''
    This class provides API to bind input/output to a specified device, e.g. GPU.
    '''
    def __init__(self, session):
        self._iobinding = C.SessionIOBinding(session._sess)

    def bind_cpu_input(self, name, arr_on_cpu):
        '''
        bind an input to array on CPU
        :param name: input name
        :param arr_on_cpu: input values as a python array on CPU
        '''
        self._iobinding.bind_input(name, arr_on_cpu)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        '''
        :param name: input name
        :param device_type: e.g. CPU, CUDA
        :param device_id: device id, e.g. 0
        :param element_type: input element type
        :param shape: input shape
        :param buffer_ptr: memory pointer to input data
        '''
        self._iobinding.bind_input(name,
                                   C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                               device_id),
                                   element_type, shape, buffer_ptr)

    def bind_output(self, name, device_type='cpu', device_id=0, element_type=None, shape=None, buffer_ptr=None):
        '''
        :param name: output name
        :param device_type: e.g. CPU, CUDA, CPU by default
        :param device_id: device id, e.g. 0
        :param element_type: output element type
        :param shape: output shape
        :param buffer_ptr: memory pointer to output data
        '''

        # Follow the `if` path when the user has not provided any pre-allocated buffer but still
        # would like to bind an output to a specific device (e.g. cuda).
        # Pre-allocating an output buffer may not be an option for the user as :
        # (1) They may not want to use a custom allocator specific to the device they want to bind the output to,
        # in which case ORT will allocate the memory for the user
        # (2) The output has a dynamic shape and hence the size of the buffer may not be fixed across runs
        if buffer_ptr is None:
            self._iobinding.bind_output(name,
                                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                                    device_id))
        else:
            if element_type is None or shape is None:
                raise ValueError("`element_type` and `shape` are to be provided if pre-allocated memory is provided")
            self._iobinding.bind_output(name,
                                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                                    device_id),
                                        element_type, shape, buffer_ptr)

    def copy_outputs_to_cpu(self):
        '''Copy output contents to CPU (if on another device). No-op if already on the CPU.'''
        return self._iobinding.copy_outputs_to_cpu()

    def clear_binding_inputs(self):
        self._iobinding.clear_binding_inputs()

    def clear_binding_outputs(self):
        self._iobinding.clear_binding_outputs()
