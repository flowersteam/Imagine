import pygame
from src.playground_env.color_generation import sample_color
from src.playground_env.env_params import *
from PIL import Image

IMAGE_PATH = '../../playground_env/icons/'
# IMAGE_PATH = './icons/'


def get_pixel_coordinates(xpos, ypos):
    return ((xpos + 1) / 2 * (SCREEN_SIZE * 2 / 3) + 1 / 6 * SCREEN_SIZE).astype(np.int), \
           ((-ypos + 1) / 2 * (SCREEN_SIZE * 2 / 3) + 1 / 6 * SCREEN_SIZE).astype(np.int)


def get_obj_identifier(object_type, color, shade, size):
    type_id = str(things.index(object_type))
    if len(type_id) == 1:
        type_id = '0' + type_id

    color_id = str(thing_colors.index(color))
    shade_id = str(thing_shades.index(shade))
    size_id = str(thing_sizes.index(size))
    return type_id + color_id + shade_id + size_id


class Thing:
    def __init__(self, color, shade, size, object_id, objects, render_mode):

        assert color in thing_colors
        assert shade in thing_shades
        assert size in thing_sizes

        self.id = object_id
        self.color = color
        self.shade = shade
        self.size = size
        self.render_mode = False  # render_mode
        self.attributes = []
        if 'absolute location' in ATTRIBUTE_LIST:
            self.absolute_location = ['center', 'center']
            self.attributes = self.absolute_location
        if 'categories' in ATTRIBUTE_LIST:
            self.attributes += ['thing']
        self.categories = ['movable', 'touchable']
        self.relative_attributes = []
        self.features = []

        self.type = None
        self.type_encoding = None
        self.position = None
        self.size_encoding = None
        self.size_pixels = None
        self.rgb_encoding = None
        self.icon = None
        self.objects = None

        self.touched = False
        self.grasped = False

        self.sample_position(objects)
        self.sample_size()
        self.sample_color()
        self.initial_rgb_encoding = self.rgb_encoding.copy()

        # rendering
        self.view = False
        self.patch = None

    def give_ref_to_obj_list(self, objects):
        self.objects = objects

    def sample_color(self):
        self.rgb_encoding = sample_color(color=self.color,
                                         shade=self.shade)

    def update_relative_attributes(self):
        self.relative_attributes = []
        if self.objects is not None:
            if 'relative_shade' in ATTRIBUTE_LIST:
                self.update_relative_shade_attribute()
            if 'relative_size' in ATTRIBUTE_LIST:
                self.update_relative_size_attribute()
        if 'relative_location' in ATTRIBUTE_LIST:
            self.update_relative_location_attributes()

    def update_attributes(self):
        if 'absolute_location' in ATTRIBUTE_LIST:
            self.update_absolute_location_attributes(self.position)
        if 'color' in ATTRIBUTE_LIST:
            self.update_color_attributes()
        if 'size' in ATTRIBUTE_LIST:
            self.update_size_attribute()

    def update_all_attributes(self):
        self.update_attributes()
        self.update_relative_attributes()

    def update_size_attribute(self):
        if self.size not in self.attributes:
            self.attributes.append(self.size)

    def update_color_attributes(self, old_color=None, old_shade=None):
        if self.color not in self.attributes:
            self.attributes.append(self.color)
            if old_color is not None:
                self.attributes.remove(old_color)
        if 'shade' in ATTRIBUTE_LIST:
            if self.shade not in self.attributes:
                self.attributes.append(self.shade)
                if old_shade is not None:
                    self.attributes.remove(old_shade)

    def update_relative_location_attributes(self):
        if self.objects is not None:
            for obj in self.objects:
                if not (obj is self):
                    if np.linalg.norm(self.position - obj.position) < NEXT_TO_EPSILON:
                        for att in obj.attributes:
                            if att not in ['left', 'right', 'top', 'bottom']:
                                self.relative_attributes.append('next_to_' + att)


    def update_absolute_location_attributes(self, new_position):
        # update absolute geo_location of objects
        if new_position[0] < 0:
            self.attributes[0] = 'left'
        else:
            self.attributes[0] = 'right'

        if new_position[1] < 0:
            self.attributes[1] = 'bottom'
        else:
            self.attributes[1] = 'top'

    def update_relative_shade_attribute(self):

        # update lightest / darkest attributes
        self.darkest = True
        self.lightest = True
        for obj in self.objects:
            if self.darkest or self.lightest and not (obj is self):
                if obj.rgb_encoding.mean() < self.rgb_encoding.mean():
                    self.darkest = False
                elif obj.rgb_encoding.mean() > self.rgb_encoding.mean():
                    self.lightest = False
        if self.lightest:
            self.relative_attributes.append('lightest')
        elif self.darkest:
            self.relative_attributes.append('darkest')

    def update_relative_size_attribute(self):
        # update biggest / smallest
        self.biggest = True
        self.smallest = True
        for obj in self.objects:
            if self.biggest or self.smallest and not (obj is self):
                if obj.size_encoding < self.size_encoding:
                    self.smallest = False
                elif obj.size_encoding > self.size_encoding:
                    self.biggest = False
        if self.smallest:
            self.relative_attributes.append('smallest')
        elif self.biggest:
            self.relative_attributes.append('biggest')

        stop = 1

    def update_position(self, new_position):
        if 'absolute_location' in ATTRIBUTE_LIST:
            self.update_absolute_location_attributes(new_position)
        self.update_relative_attributes()

        # update relative attributes
        self.position = new_position.copy()

    def get_type_encoding(self, object_type):
        self.type_encoding = np.zeros([n_things])
        self.type_encoding[things.index(object_type)] = 1

    def sample_size(self):
        if self.size == 'small':
            self.size_encoding = np.random.uniform(min_max_sizes[0][0], min_max_sizes[0][1])
        elif self.size == 'big':
            self.size_encoding = np.random.uniform(min_max_sizes[1][0], min_max_sizes[1][1])
        else:
            raise NotImplementedError
        self.size_pixels = int(RATIO_SIZE * self.size_encoding)

    def sample_position(self, objects):
        ok = False
        while not ok:
            candidate_position = np.random.uniform([-1.2, -1.2], [1.2, 1.2])
            ok = True
            for obj in objects:
                if np.linalg.norm(obj.position - candidate_position) < EPSILON:
                    ok = False
            if ok:
                self.update_position(candidate_position)

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):

        update_object_grasped = object_grasped

        # GRASP object
        # if the hand is close enough
        if np.linalg.norm(self.position - hand_position) < (self.size_encoding + GRIPPER_SIZE) / 2:
            if not self.touched and self.render_mode:
                print('Touched :', self)
            self.touched = True

            # if an object is grasped
            if object_grasped:
                # check if it's that one, if it's still grasped
                if self.grasped:
                    if not gripper_state:
                        update_object_grasped = False
                        self.grasped = False
            # if not object is grasped, check if this one is being grasped
            else:
                if gripper_state:
                    self.grasped = True
                    update_object_grasped = True
                    if self.render_mode:
                        print('Grasped :', self)

        else:
            self.touched = False

        # if grasped, the object follows the hand
        if self.grasped:
            self.update_position(hand_position.copy())
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        return update_object_grasped

    def update_color(self, new_color, new_shade, new_rgb):
        old_color = self.color
        old_shade = self.shade
        self.color = new_color
        self.shade = new_shade
        self.rgb_encoding = new_rgb
        self.update_color_attributes(old_color, old_shade)
        if self.objects:
            for obj in self.objects:
                obj.update_relative_attributes()

    def update_size(self, new_size):
        self.size_encoding = new_size
        self.size_pixels = int(RATIO_SIZE * self.size_encoding)

    def _color_surface(self, surface, rgb):

        arr = pygame.surfarray.pixels3d(surface)
        arr[:, :, 0] = rgb[0]
        arr[:, :, 1] = rgb[1]
        arr[:, :, 2] = rgb[2]



    def update_rendering(self, viewer):
        x, y = get_pixel_coordinates(self.position[0], self.position[1])
        left = int(x - self.size_pixels // 2)
        top = int(y - self.size_pixels // 2)

        color = tuple(self.rgb_encoding * 255)
        # pygame.draw.rect(self.icon, color, (0, 0, self.size_pixels - 1, self.size_pixels - 1), 6)
        self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()
        self.surface = self.icon.copy()
        self._color_surface(self.surface, color)
        viewer.blit(self.surface,(left,top))
        # viewer.blit(self.icon, (left, top))

    def __repr__(self):
        return 'Object # {}: {} {} {} {}'.format(self.id, self.size, self.shade, self.color, self.type)





class LivingThings(Thing):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        if 'category' in ATTRIBUTE_LIST:
            self.attributes += ['living_thing']
        self.categories += ['combinable']
        self.counter_food = 0
        self.counter_water = 0
        self.size_update = 0.04
        self.size_ind = 0 if self.size == 'small' else 1
        self.talked = False
        self.pet = False

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        grasped = super().update_state(hand_position, gripper_state, objects, object_grasped, action)

        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        return grasped


class Animals(LivingThings):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        if 'category' in ATTRIBUTE_LIST:
            self.attributes += ['animal']
        self.freeze = False

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        grasped = super().update_state(hand_position, gripper_state, objects, object_grasped, action)
        # check whether water or food is close
        for obj in objects:
            if obj.type == 'water' or obj.type == 'food':
                # check distance
                if np.linalg.norm(obj.position - self.position) < (self.size_encoding + obj.size_encoding) / 2:
                    # check action
                    size_encoding = min(self.size_encoding + self.size_update, min_max_sizes[1][1] + self.size_update)
                    if self.render_mode:
                        print(obj.type, ' over:', self)
                        print('Growing Living Thing {}:'.format(size_encoding), self)
                    self.update_size(size_encoding)

        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        return grasped


class Furnitures(Thing):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        if 'category' in ATTRIBUTE_LIST:
            self.attributes += ['furniture']

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        grasped = super().update_state(hand_position, gripper_state, objects, object_grasped, action)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        return grasped


class Plants(LivingThings):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        if 'category' in ATTRIBUTE_LIST:
            self.attributes += ['plant']

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        grasped = super().update_state(hand_position, gripper_state, objects, object_grasped, action)
        # check whether water or food is close
        for obj in objects:
            if obj.type == 'water':
                # check distance
                if np.linalg.norm(obj.position - self.position) < (self.size_encoding + obj.size_encoding) / 2:
                    # check action
                    size_encoding = min(self.size_encoding + self.size_update, min_max_sizes[1][1] + self.size_update)
                    if self.render_mode:
                        print(obj.type, ' over:', self)
                        print('Growing Living Thing {}:'.format(size_encoding), self)
                    self.update_size(size_encoding)

        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        return grasped


class Supplies(Thing):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        if 'category' in ATTRIBUTE_LIST:
            self.attributes += ['supply']
        self.categories += ['combinable']
        self.grasped = False



# # # # # # # # # # # # # # # # # #
# Animals
# # # # # # # # # # # # # # # # # #

class Dog(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'dog'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['dog']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'dog.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Cat(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'cat'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['cat']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'cat.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Human(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'human'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['human']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'human.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Fly(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'fly'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['fly']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'fly.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Parrot(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'parrot'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['parrot']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'parrot.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Mouse(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'mouse'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['mouse']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'mouse.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Lion(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'lion'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['lion']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'lion.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Pig(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'pig'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['pig']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'pig.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Cow(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'cow'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['cow']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'cow.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Cameleon(Animals):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'cameleon'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['cameleon']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'cameleon.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


# # # # # # # # # # # # # # # # # #
# Plants
# # # # # # # # # # # # # # # # # #

class Cactus(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'cactus'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['cactus']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'cactus.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Rose(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'rose'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['rose']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'rose.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Grass(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'grass'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['grass']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'grass.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Bonsai(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'bonsai'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['bonsai']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'bonsai.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Algae(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'algae'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['algae']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'algae.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Carnivorous(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'carnivorous'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['carnivorous']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'carnivorous.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Tree(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'tree'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['tree']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'tree.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Bush(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'bush'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['bush']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'bush.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Tea(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'tea'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['tea']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'tea.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Flower(Plants):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'flower'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['flower']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'flower.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


# # # # # # # # # # # # # # # # # #
# Furniture
# # # # # # # # # # # # # # # # # #

class Chair(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'chair'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['chair']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'chair.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Sofa(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'sofa'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['sofa']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'sofa.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Sink(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'sink'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['sink']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'sink.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Window(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'window'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['window']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'window.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Carpet(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'carpet'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['carpet']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'carpet.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Cupboard(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'cupboard'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['cupboard']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'cupboard.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Desk(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'desk'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['desk']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'desk.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Lamp(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'lamp'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['lamp']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'lamp.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Door(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'door'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['door']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'door.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Table(Furnitures):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'table'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['table']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'table.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        grasped = super().update_state(hand_position, gripper_state, objects, object_grasped, action)
        # check whether water or food is close
        for obj in objects:
            if obj.type == 'food':
                # check distance
                if np.linalg.norm(obj.position - self.position) < (self.size_encoding + obj.size_encoding) / 2:
                    # check action
                    self.rgb_encoding = np.zeros([3])
                    assert not np.all(self.rgb_encoding == self.initial_rgb_encoding)
                    if self.render_mode:
                        print(obj.type, ' over:', self)
                        print('Put food on the table.', self)

        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        return grasped


# # # # # # # # # # # # # # # # # #
# Supply
# # # # # # # # # # # # # # # # # #


class Water(Supplies):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'water'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['water']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'water.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


class Food(Supplies):
    def __init__(self, color, shade, size, object_id, objects, render_mode):
        super().__init__(color, shade, size, object_id, objects, render_mode)
        self.type = 'food'
        self.get_type_encoding(self.type)
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        self.features = np.concatenate(
            [self.type_encoding, self.position, np.array([self.size_encoding]), self.rgb_encoding, grasped_feature])
        if 'type' in ATTRIBUTE_LIST:
            self.attributes += ['food']
        self.obj_identifier = get_obj_identifier(self.type, color, shade, size)
        if render_mode:
            self.icon = pygame.image.load(IMAGE_PATH + 'food.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)).convert_alpha()


if 'small' in ENV_ID:
    things_classes = [Dog, Cat, Cameleon, Human, Fly,
                      Cactus, Carnivorous, Flower, Tree, Bush,
                      Door, Chair, Desk, Lamp, Table,
                      Food, Water]

elif 'big' in ENV_ID:
    things_classes = [Dog, Cat, Cameleon, Human, Fly, Parrot, Mouse, Lion, Pig, Cow,
                      Cactus, Carnivorous, Flower, Tree, Bush, Grass, Algae, Tea, Rose, Bonsai,
                      Door, Chair, Desk, Lamp, Table, Cupboard, Sink, Window, Sofa, Carpet,
                      Food, Water]

else:
    raise NotImplementedError


#

def build_object(object_type, color, shade, size, object_id, objects, render_mode):
    assert object_type in things
    obj_class = things_classes[things.index(object_type)](color, shade, size, object_id, objects, render_mode)
    assert obj_class.type == object_type, '{}, {}'.format(obj_class.type, object_type)
    return obj_class


stop = 1
