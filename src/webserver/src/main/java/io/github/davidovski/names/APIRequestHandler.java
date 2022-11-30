package io.github.davidovski.names;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

@SuppressWarnings("restriction")
public class APIRequestHandler implements HttpHandler {

    private NameDatabaseManager nameDatabaseManager;

    public APIRequestHandler() {
        nameDatabaseManager = new NameDatabaseManager(new File("names.db"));
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        // get the requested path
        String path = exchange.getRequestURI().getPath();

        System.out.println(path);

        if (path.startsWith("/api/name")) {

            // parse the body as a json object
            JSONTokener jsonParser = new JSONTokener(exchange.getRequestBody());
            JSONObject json = (JSONObject) jsonParser.nextValue();

            System.out.println(json.toString(2));

            if (json == null) {
                // Malformed JSON request, 400
                sendJSON(exchange, (new JSONObject()).put("message", "Malformed JSON body"), 400);
                return;
            }

            // generate name(s) and send it
            generateName(exchange, json);
        } else {
            sendJSON(exchange, (new JSONObject()).put("message", "Not Found"), 404);
        }
    }

    public void generateName(HttpExchange exchange, JSONObject options) throws JSONException, IOException {
        String origin = options.optString("origin", "none").toLowerCase();

        int count = options.optInt("count", 1);

        // ensure that the count is between 1-100
        if (count < 1 || count > 100) {
            sendJSON(exchange, (new JSONObject()).put("message", "Name count is out of range: Ensure that the request is between 1 and 100 names"), 400);
            return;
        }

        String gender = options.optString("gender", "female");

        // ensure that the gender is either male or female
        if (!gender.equals("male") && !gender.equals("female")) {
            sendJSON(exchange, (new JSONObject()).put("message", "Requested gender is invalid"), 400);
            return;
        }

        // Store the names in an array
        List<String> names = nameDatabaseManager.getRandomNames(origin, gender, count);

        if (options.optBoolean("surname")) {
            List<String> surnames = nameDatabaseManager.getRandomNames(origin, "surname", count);

            // Add surnames to the end of each firstname in the names list
            for (int i = 0; i < count; i++) {
                String fullname = names.get(i) + " " + surnames.get(i);
                names.set(i, fullname);
            }
        }

        // Create the response json object
        JSONObject response = new JSONObject();
        response.put("message", "Generated " + count + " names");
        response.put("names", new JSONArray(names));

        // send the json back to the client
        sendJSON(exchange, response, 200);
    }

    public void sendJSON(HttpExchange exchange, JSONObject json, int responseCode) throws IOException {
        // convert the json to a string
        String response = json.toString(2);

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

}
