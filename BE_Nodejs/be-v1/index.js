const express = require('express')
require("dotenv").config()
const cors = require("cors"); // import cors
const bodyParser = require('body-parser'); // import body-parser
const clientRoutes = require("./routes/client/index.route") // import client routes
// const adminRoutes = require("./routes/admin/index.route") // import admin routes

//Connect db mongoose
const database = require("./config/database")
database.connect();



const app = express()
const port = process.env.PORT



//fix cors fe
app.use(cors())

// body-parser
app.use(bodyParser.json()); 
app.use(bodyParser.urlencoded({ extended: true }));

//Routes
clientRoutes(app)
// adminRoutes(app)


app.listen(port)