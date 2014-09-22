# happening README

## Purpose

Aggregate streaming social activity to find events that you might not otherwise stumble upon!

Not just a real-time social media mapper, find anomalous events that stand out above typical background activity!

That is to say: Locate trending social activity in meatspace!

## Getting Started

### Dependencies

* twitter: A twitter account, an application [dev.twitter.com/apps](https://dev.twitter.com/apps), and its authentication keys.

* tweepy: `pip install tweepy`

### Keys

Make a file named `twitauth.cfg` in the project's root directory your application's keys from [dev.twitter.com](https://dev.twitter.com/).

```
apikey=
apikeysecret=
accesstoken=
accesstokensecret=
```

### Stream data from a geographic region

```
python streamFromGeo_twitter.py
```

Currently writes data to disk.

### Search for data within a geographic region

```
python searchForGeo_twitter.py
```

Currently does not write data to disk.

# Running the app

2. **Recommended:** Install virtualenv and fire up a virtual environment.

  ```
  # Install virtualenv
  sudo pip install virtualenv

  # Create virtualenv folder `venv`
  virtualenv venv

  # Activate the virtual environment
  source venv/bin/activate
  ```

3. Install Python project dependencies.

  ```
  pip install -r requirements.txt
  ```

4. To test your application, run the manage.py file: `python manage.py runserver`, and open your web browser to
`localhost:5000`.
