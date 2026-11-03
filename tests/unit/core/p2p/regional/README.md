# Regional Signalling Client Tests

Tests of the client's regional signalling mode. The fakes in `conftest.py` play
the backend token route and one region.

## Contract

`neuracore.core.streaming.p2p.signalling_session`:

| Name | Behaviour |
|---|---|
| `SignallingSession(api_url, org_id, auth, client_session, robot_id=None, robot_instance=None, clock=...)` | Token holder for one robot instance or one user. |
| `.headers()` | `Authorization` with the current token. Gets a token if none, renews after `renew_after`. Never sends `register_epoch`. |
| `.register(base_url, stream_id)` | Gets a token with `register_epoch: true`, then posts the registration. Returns `RegistrationResult(region, canonical_url, epoch)`. |
| `.region_endpoints` | `region_endpoints` of the last token response. |
| `LowerEpochError` | Raised by `register` on 409 `lower_epoch`. |

Other changes:

- `BaseSSEConsumer.on_close(reason)`: called on an SSE `close` event. The consumer connects again. An `end` event still stops it.
- `BaseSSEConsumer`: every reconnect goes through the consumer loop, with a new config. `on_reconnect` runs before reconnecting after a stream that opened, until it succeeds.
- `SignallingEventsConsumer.signalling_session`: in regional mode, the stream and heartbeats go to the session's region with the session's token. A reconnect, or a `resurrected` heartbeat, makes the robot register again with a new epoch. On `lower_epoch`, the region holds the stream at a newer epoch: the consumer stops.
- A 401 or 403 from the backend or the region raises `AuthenticationError`, which ends the consumer.
