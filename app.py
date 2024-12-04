from flask import Flask, render_template, request
import your_optimization_module  # Import your code for song optimization
import os

app = Flask(__name__)

# Dummy song data
songs = [
    {"id": 1, "name": "Song 1"},
    {"id": 2, "name": "Song 2"},
    {"id": 3, "name": "Song 3"},
]

# Create the output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Render the home page to a static file
with app.test_request_context():
    with open('output/index.html', 'w') as f:
        f.write(render_template('index.html', songs=songs))

    with open('output/result.html', 'w') as f:
        f.write(render_template('result.html', song=songs[0], best_feature="Feature A", optimal_value=0.8, max_increase=0.15))



@app.route('/')
def home():
    return render_template('index.html', songs=songs)

@app.route('/optimize', methods=['POST'])
def optimize():
    song_id = int(request.form['song_id'])
    song = next(song for song in songs if song["id"] == song_id)
    
    

    best_feature, optimal_value, max_increase = your_optimization_module.optimize_feature_for_song(song)
    return render_template('result.html', song=song, best_feature=best_feature, optimal_value=optimal_value, max_increase=max_increase)

if __name__ == '__main__':
    app.run(debug=True)