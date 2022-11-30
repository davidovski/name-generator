package io.github.davidovski.names;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

@SuppressWarnings("restriction")
public class StaticRequestHandler implements HttpHandler {
    private File root;

    public StaticRequestHandler(File root) {
        this.root = root;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        // get the requested path
        String path = exchange.getRequestURI().getPath();

        File requested = new File(root, path);

        // automatically send the index of a directory
        if (requested.isDirectory()) {
            requested = new File(requested, "index.html");
        }

        // ensure that the file is in the intended document root
        if (!isInRoot(requested, root)) {
            sendText(exchange, "Access Denied", 403);
        } else if (requested.exists()) {
            sendFile(exchange, requested);
        } else {
            // send 404 if the file isnt found
            sendText(exchange, "File Not Found", 404);
        }
    }

    private void sendFile(HttpExchange exchange, File file) throws IOException {
        // read the file as into an array of bytes
        byte[] bytes = Files.readAllBytes(file.toPath());

        // send the file headers
        exchange.sendResponseHeaders(200, bytes.length);

        // send the file body
        OutputStream os = exchange.getResponseBody();
        os.write(bytes);
        os.close();
    }

    private void sendText(HttpExchange exchange, String response, int responseCode) throws IOException {
        // calculate the response content size
        int contentSize = response.toString().getBytes().length;

        // set the response headers
        exchange.getResponseHeaders().add("Content-Type", "text/json");
        exchange.sendResponseHeaders(responseCode, contentSize);

        // write the response to the output stream
        OutputStream outputStream = exchange.getResponseBody();

        outputStream.write(response.toString().getBytes());
        outputStream.close();
    }

    public static boolean isInRoot(File request, File root) {
        File parentFile = request;

        // start from the requested file and traverse upwards until reaching the root directory
        while (parentFile != null) {
            if (root.equals(parentFile)) {
                return true;
            }
            parentFile = parentFile.getParentFile();
        }

        // If there isn't a parent file, return false
        return false;
    }

}
