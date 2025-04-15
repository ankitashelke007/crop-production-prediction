
@app.route("/yield1", methods=["GET", "POST"])
def yield1():
    # existing yield prediction logic here...


    if request.method == "POST":
        try:
            dist_name = request.form["dist_name"].strip().lower()
            crop = request.form["crop"].strip().lower()
