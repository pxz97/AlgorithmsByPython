class BaseModel:
    def __init__(self):
        super(BaseModel, self).__init__()
        self.genre_vocab = ["Film-Noir", "Action", "Adventure", "Horror", "Romance", "War", "Comedy", "Western",
                            "Documentary", "Sci-Fi", "Drama", "Thriller", "Crime", "Fantasy", "Animation", "IMAX",
                            "Mystery", "Children", "Musical"]

        self.GENRE_FEATURES = {
            "userGenre1": self.genre_vocab,
            "userGenre2": self.genre_vocab,
            "userGenre3": self.genre_vocab,
            "userGenre4": self.genre_vocab,
            "userGenre5": self.genre_vocab,
            "movieGenre1": self.genre_vocab,
            "movieGenre2": self.genre_vocab,
            "movieGenre3": self.genre_vocab,
        }

    def build(self):
        pass
