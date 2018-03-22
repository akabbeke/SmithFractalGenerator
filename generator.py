
import numpy
import random
import uuid

import json
import os

random.seed()

from datetime import datetime
from PIL import Image
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

with open(os.path.join(os.path.dirname(__file__), "config.json"), 'r') as config_file:
    CONFIG = json.load(config_file)

class Background:
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        self.image_upscale_factor = kwargs.get("image_upscale_factor")
        self.board_seed_factor = kwargs.get("board_seed_factor")
        self.subshape_seed_factor = kwargs.get("subshape_seed_factor")
        self.step_count = kwargs.get("step_count")
        self.step_scale = kwargs.get("step_scale")
        self.conway_cycles = kwargs.get("conway_cycles")
        self.conway_breed_factor = kwargs.get("conway_breed_factor")
        self.conway_live_factor = kwargs.get("conway_live_factor")
        self.density_threshold = kwargs.get("density_threshold")
        self.sd_threshold = kwargs.get("sd_threshold")
        self.image = self.load_image()
        self.step_count = self.recalculate_step_count()
        self.subshape = self.generate_subshape()
        self.board = self.seed_board()


    def load_image(self):
        image = Image.open(self.filepath)
        shape = image.size
        return image.resize(
            [
                shape[0]*self.image_upscale_factor,
                shape[1]*self.image_upscale_factor
            ],
            Image.ANTIALIAS
        )


    def generate_subshape(self):
        return numpy.random.rand(
            self.step_scale,
            self.step_scale
        ) > self.subshape_seed_factor


    def recalculate_step_count(self):
        count = self.step_count
        while self.step_scale**count > self.image.size[0] or self.step_scale**count > self.image.size[1]:
            count -= 1
        return count


    @property
    def seed_shape(self):
        return [
            int(self.image.size[0]/self.growth_factor),
            int(self.image.size[1]/self.growth_factor)
        ]


    @property
    def growth_factor(self):
        return self.step_scale**self.step_count


    @property
    def output_file(self):
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        return os.path.join(CONFIG["output_dir"], "{}.png".format(filename))

    @property
    def fail_file(self):
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        return os.path.join(CONFIG["fail_dir"], "{}_{}_{}_{}.png".format(
            filename,
            self.step_count,
            self.conway_cycles,
            self.conway_breed_factor,
            self.conway_live_factor
        ))

    def seed_board(self):
        return numpy.random.rand(*self.seed_shape) > self.board_seed_factor


    def life_step(self):
        board = numpy.asarray(self.board)
        assert board.ndim == 2
        board = board.astype(bool)
        board_sum = sum(
            numpy.roll(numpy.roll(board, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0)
        )
        return ((board_sum == self.conway_live_factor) & board) | (board_sum == self.conway_breed_factor)


    def tessellate(self):
        new_board = numpy.zeros([
            self.board.shape[0]*self.step_scale,
            self.board.shape[1]*self.step_scale
        ])
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                x_lower = i*self.step_scale
                x_upper = (i+1)*self.step_scale
                y_lower = j*self.step_scale
                y_upper = (j+1)*self.step_scale
                new_board[x_lower:x_upper, y_lower:y_upper] = self.board[i,j]*self.subshape
        return new_board

    def generate_fractal(self):
        for cycle_count in range(self.step_count+1):
            if self.is_dead():
                return
            for i in range(self.conway_cycles):
                self.board = self.life_step()
            if cycle_count != self.step_count:
                self.board = self.tessellate()

    def is_dead(self):
        return numpy.sum(self.board) == 0

    def is_bad(self):
        average = numpy.sum(self.board) / (1.0 * self.board.size)
        return average < 0.01 or average > 0.20

    def is_line_type(self):
        not_x = False
        not_y = False
        for i in range(self.board.shape[0]):
            if numpy.sum(self.board[i,:]) not in [0, self.board.shape[1]]:
                not_x = True
        for i in range(self.board.shape[1]):
            if numpy.sum(self.board[:,i]) not in [0, self.board.shape[0]]:
                not_y = True
        if not (not_x or not_y):
            print "line type"
            print self.board
        return not (not_x or not_y)


    def is_space_filling(self):
        average = numpy.sum(self.board) / (1.0 * self.board.size)
        densities = []
        for i in range(1000):
            subsample = self.subsample()
            densities += [numpy.sum(subsample) / (1.0 * subsample.size)]
        standard_deviation = (sum([(d-average)**2 for d in densities])/len(densities))**0.5
        return average > self.density_threshold and standard_deviation < self.sd_threshold

    def is_success(self):
        return not self.is_bad() and not self.is_line_type() and not self.is_space_filling()

    def subsample(self):
        size = 100
        pos_x = random.choice(range(0, self.board.shape[0]-size-1))
        pos_y = random.choice(range(0, self.board.shape[1]-size-1))
        return self.board[pos_x:pos_x+size, pos_y:pos_y+size]

    def save_success(self):
        self.save_image(self.output_file)
    
    def save_failure(self):
        self.save_image(self.fail_file)

    def save_image(self, path):
        output_image = Image.new('RGBA', self.board.shape)
        output_pixels = output_image.load()
        input_pixels = self.image.load()
        for i in range(output_image.size[0]):
            for j in range(output_image.size[1]):
                if self.board[i,j] == 1:
                    output_pixels[i,j] = input_pixels[i,j]
                else:
                    output_pixels[i,j] = (0, 0, 0)
        output_image.save(path)

def random_config():
    return {
        "image_upscale_factor": CONFIG["image_upscale_factor"],
        "board_seed_factor": random.choice(CONFIG["board_seed_factor_options"]),
        "subshape_seed_factor": random.choice(CONFIG["subshape_seed_factor_options"]),
        "step_count": random.choice(CONFIG["step_count_options"]),
        "step_scale": random.choice(CONFIG["step_scale_options"]),
        "conway_cycles": random.choice(CONFIG["conway_cycles_options"]),
        "conway_breed_factor": random.choice(CONFIG["conway_breed_factor_options"]),
        "conway_live_factor": random.choice(CONFIG["conway_live_factor_options"]),
        "density_threshold": random.choice(CONFIG["density_threshold_options"]),
        "sd_threshold": random.choice(CONFIG["sd_threshold_options"])
    }

def backgroundize(path):
    success = False
    while not success:
        background = Background(path, **random_config())
        background.generate_fractal()
        if background.is_success():
            success = True
            background.save_success()
        else:
            if not background.is_dead():
                background.save_failure()

def main():
    pool = Pool(16)
    paths = [join(CONFIG["input_dir"], f) for f in listdir(CONFIG["input_dir"]) if isfile(join(CONFIG["input_dir"], f)) and f.endswith('.jpg')]
    pool.map(backgroundize, paths)

if __name__ == "__main__":
    main()


