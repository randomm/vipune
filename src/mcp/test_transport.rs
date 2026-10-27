//! A minimal no-op MCP transport, so a test can build the `RequestContext`
//! that the public `ServerHandler::call_tool` entry point requires.
//!
//! `rmcp`'s `Peer::new` is `pub(crate)`, so the only supported way to obtain a
//! `Peer` outside the crate is through `serve_directly`, which needs a
//! `Transport`. vipune's tool handlers take only `Parameters<T>` and never
//! touch the request context, so a transport that receives nothing and sends
//! nowhere is sufficient — and it keeps the test on the same `call_tool` path a
//! real stdio client goes through.

use rmcp::ServerHandler;
use rmcp::model::{
    ClientCapabilities, Implementation, InitializeRequestParams, ProtocolVersion,
    ServerCapabilities, ServerConfig,
};
use rmcp::service::{
    RequestContext, RoleServer, RxJsonRpcMessage, TxJsonRpcMessage, serve_directly,
};
use rmcp::transport::Transport;
use tokio::sync::mpsc;

/// The transport handed to [`serve_directly`]: it receives nothing (the inbound
/// sender is dropped immediately) and sends nowhere (the outbound receiver is
/// dropped immediately).
struct NullServerTransport {
    rx: mpsc::Receiver<RxJsonRpcMessage<RoleServer>>,
    tx: mpsc::Sender<TxJsonRpcMessage<RoleServer>>,
}

impl Transport<RoleServer> for NullServerTransport {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let tx = self.tx.clone();
        async move {
            // Nobody reads the outbound side; the test calls the tools
            // directly, so an undeliverable message is not a failure.
            let _ = tx.try_send(item);
            Ok(())
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleServer>>> + Send {
        self.rx.recv()
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.rx.close();
        async { Ok(()) }
    }
}

/// Placeholder service for [`serve_directly`]; its handlers are never reached
/// because the inbound stream is closed and the service is cancelled at once.
#[derive(Clone)]
struct NullHandler;

impl ServerHandler for NullHandler {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().build())
    }
}

/// Build a [`RequestContext`] suitable for a direct `call_tool` invocation.
pub(crate) fn test_request_context() -> RequestContext<RoleServer> {
    let (inbound_producer, inbound) = mpsc::channel::<RxJsonRpcMessage<RoleServer>>(1);
    drop(inbound_producer);
    let (outbound, outbound_reader) = mpsc::channel::<TxJsonRpcMessage<RoleServer>>(1);
    drop(outbound_reader);

    let running = serve_directly(
        NullHandler,
        NullServerTransport {
            rx: inbound,
            tx: outbound,
        },
        Some(
            InitializeRequestParams::new(
                ClientCapabilities::default(),
                Implementation::new("vipune-test", env!("CARGO_PKG_VERSION")),
            )
            .with_protocol_version(ProtocolVersion::LATEST),
        ),
    );
    let peer = running.peer().clone();
    // Cancel the service task: nothing will ever be served on this transport,
    // and the returned future only needs to be dropped rather than awaited.
    drop(running.cancel());
    RequestContext::new(rmcp::model::RequestId::Number(1), peer)
}
