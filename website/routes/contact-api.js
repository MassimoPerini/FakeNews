/**
 * Created by massimo on 29/06/17.
 */

var debug = require('debug')('api');
var express = require('express');

var router = express.Router();

/**
 * @swagger
 * /contact:
 *   post:
 *     tags:
 *       - Contact
 *     summary: Create an inquiry
 *     description: used to send a message to the clinic.
 *     produces:
 *      - application/json
 *     parameters:
 *       - in: body
 *         name: message
 *         description: the message inquiry to create
 *         required: true
 *         schema:
 *           $ref: '#/definitions/Inquiry'
 *     responses:
 *       200:
 *         description: Succesful operation.
 *         examples:
 *           application/json:
 *             message: Thank you!
 *         schema:
 *           $ref: '#/responses/GenericMessage'
 *       400:
 *         description: Error case.
 *         schema:
 *           $ref: '#/responses/GenericMessage'
 *         examples:
 *           application/json:
 *             message: error
 */
router.post('/api/contact', function(req, res, next) {
    /*
    models.inquiries.create({
        name: req.body.name,
        surname: req.body.surname,
        phone: req.body.phone,
        mail: req.body.mail,
        target: req.body.target,
        obj: req.body.object,
        message: req.body.message
    }).then(function () {
        return res.status(200).json({message: "Thank you!"});
    }).catch(function (err) {
        debug(err);
        return res.status(400).json({ message: "error" });
    });
    */
});

/**
 * @swagger
 * responses:
 *   GenericMessage:
 *     description: Generic default answer message
 *     required:
 *       - message
 *     properties:
 *       message:
 *         type: string
 */

module.exports = router;
