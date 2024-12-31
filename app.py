from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the preprocessed data and models
final_df = pickle.load(open('final.pkl', 'rb'))
pt = pickle.load(open('pivot_table.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'index.html',
        book_name=list(final_df['Title'].values),
        author=list(final_df['Author'].values),
        image=list(final_df['img_url'].values),
        votes=list(final_df['num_rating'].values),
        ratings=list(final_df['avg_rating'].values),
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    print(f"User Input: {user_input}")

    # Check if the book exists
    indices = np.where(pt.index == user_input)[0]
    if len(indices) == 0:
        return render_template('recommend.html', error="Book not found!")

    book_id = indices[0]
   
    # Get the 5 nearest neighbors
    distances, suggestions = model.kneighbors(pt.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    data = []
    for i in suggestions[0][1:]:  # Skip the first suggestion as it's the input book itself
        temp_df = books[books['Title'] == pt.index[i]]
        item = [
            temp_df['Title'].iloc[0],
            temp_df['Author'].iloc[0],
            temp_df['img_url'].iloc[0],
        ]
        data.append(item)

    print("Recommendations:", data)
    return render_template('recommend.html', data=data)

@app.route('/contact', methods=['GET','POST'])
def contact():
    name = email = message = None
    if request.method == 'POST':
       name = request.form['name']
       email = request.form['email']
       message = request.form['message']

    # For simplicity, you can print the data or save it to a database
       print(f"Contact Form Submitted: {name}, {email}, {message}")

       return render_template('contact.html', success="Thank you for reaching out! We'll get back to you soon.")

    return render_template('contact.html',name=name,email=email,message=message)
if __name__ == '__main__':
    app.run(debug=True)
