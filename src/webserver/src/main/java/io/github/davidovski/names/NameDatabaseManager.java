package io.github.davidovski.names;

import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class NameDatabaseManager {
    private File databaseFile;
    private Connection connection;

    private static final String TABLE = "names";

    public NameDatabaseManager(File databaseFile) {
        this.databaseFile = databaseFile;

        connect();
    }

    /**
     * Creates a connection to the database. If one could not be created, the connection will remain as null
     */
    private void connect() {
        connection = null;
        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + databaseFile.getPath());
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public List<String> getRandomNames(String origin, String category, int quantitiy) {
        // create the set to return, even if empty
        List<String> names = new ArrayList<String>();

        if (connection != null) {
            // Create an sql statement
            String sql = "SELECT Name FROM " + TABLE + " WHERE Origin=? AND Category=? ORDER BY RANDOM() LIMIT ?;";
            PreparedStatement statement;
            try {
                statement = connection.prepareStatement(sql);

                // insert relevant values into the statement
                statement.setString(1, origin);
                statement.setString(2, category);
                statement.setInt(3, quantitiy);

                // execute the query and get the result
                ResultSet result = statement.executeQuery();

                // Add all of the results to the names set
                while (result.next()) {
                    String name = result.getString("Name");
                    names.add(name);
                }
            } catch (SQLException e) {
                e.printStackTrace();
            }

        }
        return names;
    }

    public static void main(String[] args) {
        NameDatabaseManager dbManager = new NameDatabaseManager(new File("names.db"));
        List<String> names = dbManager.getRandomNames("spain", "female", 10);
        for (String name : names) {
            System.out.println(name);
        }
    }
}
