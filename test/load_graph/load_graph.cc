#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"


using namespace tensorflow;

int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "../graph_pb/graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  status = session->Run({}, {}, {"init"}, {});
  status = session->Run({},
                        {"c", "gradients/c_grad/tuple/control_dependency:0",
                         "gradients/c_grad/tuple/control_dependency_1:0"},
                        {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();
  auto output_grad_a = outputs[1].scalar<float>();
  auto output_grad_b = outputs[2].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString()
            << "\n";                // Tensor<type: float shape: [] values: 30>
  std::cout << output_c() << "\n";  // 30
  std::cout << output_grad_a() << "\n";  // 6
  std::cout << output_grad_b() << "\n";  // 5

  // Free any resources used by the session
  session->Close();
  return 0;
}
