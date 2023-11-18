const http = require("http");
const socket = require("socket.io");
const socketServer = require("./socket");

const server = http.createServer();

const socketIo = socket(server, {
  transports: ["websocket", "polling"],
  cors: {
    origin: "*",
  },
});

socketServer(socketIo);
server.listen(3000, () => {
  console.log("Server listening on port 3000");
});
