const getConfig = () => {};

module.exports = async (socketIo) => {
  socketIo.on("connection", (client) => {
    console.log("User connected: ", client.id);

    client.on("config", (data) => {
      console.log(data);
      socketIo.emit("config", { data: { coordinates: data } });
    });

    client.on("live-coordinates", (data) => {
      console.log(data);
      socketIo.emit("live-coordinates", { data: { coordinate: data } });
    });

    client.on("disconnect", () => {
      console.log("User disconnected: ", client.id);
    });
  });
};
