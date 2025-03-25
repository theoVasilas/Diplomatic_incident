from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context


def create_server_app(cfg, strategy):
        
        def server_fn(context: Context) -> ServerAppComponents:
            config = ServerConfig(num_rounds = cfg.ROUNDS)
            print(f"=== {type(strategy).__name__} ==== \n")
            return ServerAppComponents(strategy=strategy, config=config)

        server = ServerApp(server_fn=server_fn)
        return server