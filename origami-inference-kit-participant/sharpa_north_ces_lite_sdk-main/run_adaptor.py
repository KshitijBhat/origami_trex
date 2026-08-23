import numpy as np
from examples.remote_observation_client import RemoteObservationClient
from vitac_policy_server import TeamPolicy

adapter = TeamPolicy(25)

with RemoteObservationClient(
    endpoint='tls/challenge.sharpa.com.cn:7448',
    session_id='OrVizKar',
    token='iros_977dcb61d7a495430cb8ac8e67ebd5e0ae8bd5f65f2c233e49c3be4c27ed4bf8',
    tls_root_ca_certificate='/etc/ssl/certs/ca-certificates.crt',
) as client:
    observation = client.get_observation()
    result = adapter.infer(observation)

actions = np.asarray(result["actions"])
assert actions.dtype == np.float32
assert actions.ndim == 2
assert actions.shape[1] == 65
assert np.isfinite(actions).all()