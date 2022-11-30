package io.github.davidovski.names;

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;

import com.sun.net.httpserver.HttpServer;

@SuppressWarnings("restriction")
public class WebServer {
    private int port;
    private HttpServer server;

    public WebServer(int port) throws IOException {
        this.port = port;

        // create an HTTP server instance
        InetSocketAddress socketAddress = new InetSocketAddress(port);
        server = HttpServer.create(socketAddress, 0);

        // create a context for the static request handler
        server.createContext("/", new StaticRequestHandler(new File("dist")));
        // create a context for the api request handler
        server.createContext("/api", new APIRequestHandler());

    }

    public void start() {
        server.start();

        // tell the user that the webserver has been started, and the port used
        System.out.println("Webserver started on port " + port);
    }


    public static void main(String[] args) throws IOException {

        // set a default port number
        int port = 8080;

        // parse the arguments for a port number
        if (args.length > 0)
            port = Integer.parseInt(args[0]);

        // Create the webserver instance
        WebServer webServer = new WebServer(port);

        // start the webserver
        webServer.start();
    }
}
