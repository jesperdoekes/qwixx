"""
Simple CLI implementation of the dice game Qwixx (simplified rules).

How to play:
- Run the script. Choose number of players and names.
- Each turn an active player rolls 6 dice: 2 white and 4 colored (R,Y,G,B).
- All players may mark the sum of the two white dice on any row (if available) by typing 'w' when prompted.
- The active player may additionally mark one of the four color+white sums (for that color).
- You can always skip a marking by pressing Enter.
- The game ends when two rows are locked or the players decide to stop.

This is a simplified implementation intended for study / PWS project.
"""
"""Qwixx (simplified) — per-player score sheets with directional rules.

Features added/implemented:
- Each player has their own score sheet.
- Red (R) and Yellow (Y) rows go 2..12 left->right.
- Green (G) and Blue (B) rows go 12..2 left->right.
- Players may only mark numbers to the right of their last mark in that row (i.e. can't go back).
- Active player may choose which white die to combine with a chosen colored die.

This is still a simplified CLI for demonstration and study.
"""

import random
from typing import List, Dict, Optional


COLORS = ['R', 'Y', 'G', 'B']
NUMBERS = list(range(2, 13))  # 2..12


def number_to_index(color: str, number: int) -> Optional[int]:
	"""Map (color, number) to row index 0..10 (left-to-right). Returns None if number invalid."""
	if number < 2 or number > 12:
		return None
	if color in ('R', 'Y'):
		return number - 2
	else:  # G, B: display 12..2 left->right
		return 12 - number


class ScoreSheet:
	def __init__(self):
		# rows: each is a list of 11 booleans (left-to-right display order)
		self.rows: Dict[str, List[bool]] = {c: [False] * 11 for c in COLORS}

	def last_mark_index(self, color: str) -> int:
		row = self.rows[color]
		# return highest index marked, or -1 if none
		for i in range(len(row) - 1, -1, -1):
			if row[i]:
				return i
		return -1

	def can_mark(self, color: str, number: int) -> bool:
		idx = number_to_index(color, number)
		if idx is None:
			return False
		row = self.rows[color]
		if row[idx]:
			return False
		last = self.last_mark_index(color)
		# Must mark to the right of last mark (strictly greater idx)
		return idx > last

	def mark(self, color: str, number: int) -> bool:
		if not self.can_mark(color, number):
			return False
		idx = number_to_index(color, number)
		assert idx is not None
		self.rows[color][idx] = True
		return True

	def marks_count(self, color: str) -> int:
		return sum(1 for m in self.rows[color] if m)

	def score(self) -> int:
		# standard Qwixx row scoring per row: n*(n+1)/2
		total = 0
		for c in COLORS:
			n = self.marks_count(c)
			total += n * (n + 1) // 2
		return total

	def __str__(self) -> str:
		# Pretty ascii table per sheet
		lines = []
		for c in ['R', 'Y', 'G', 'B']:
			if c in ('R', 'Y'):
				nums = list(range(2, 13))
			else:
				nums = list(range(12, 1, -1))
			marks = ['X' if m else '.' for m in self.rows[c]]
			row = ' '.join(f"{n:2}:{mk}" for n, mk in zip(nums, marks))
			lines.append(f"{c}: {row}")
		return '\n'.join(lines)


class Player:
	def __init__(self, name: str):
		self.name = name
		self.sheet = ScoreSheet()
		self.penalties = 0

	def add_penalty(self):
		self.penalties += 1

	def total_score(self) -> int:
		return self.sheet.score() - self.penalties * 5


class QwixxGame:
	def __init__(self, player_names: List[str]):
		if not 1 <= len(player_names) <= 6:
			raise ValueError('players must be 1-6')
		self.players: List[Player] = [Player(n) for n in player_names]
		self.active = 0
		self.round = 0

	def roll(self):
		w1 = random.randint(1, 6)
		w2 = random.randint(1, 6)
		colors = {c: random.randint(1, 6) for c in ['R', 'Y', 'G', 'B']}
		return w1, w2, colors

	def pretty_display_roll(self, w1: int, w2: int, colors: Dict[str, int]):
		white_sum = w1 + w2
		print('\n' + '=' * 60)
		print(f"Round {self.round} — Active: {self.players[self.active].name}")
		print('-' * 60)
		print(f"White dice: [1]={w1}  [2]={w2}   sum={white_sum}")
		print('Colored dice:')
		for c in ['R', 'Y', 'G', 'B']:
			print(f"  {c}: {colors[c]}   (w1+{c}={w1 + colors[c]}, w2+{c}={w2 + colors[c]})")
		print('-' * 60)

	def play_turn(self):
		number_allowed = 0
		check_penalties = 0
		self.round += 1
		w1, w2, colors = self.roll()
		white_sum = w1 + w2
		self.pretty_display_roll(w1, w2, colors)
		
		active_player = self.players[self.active]
		while number_allowed == 0:
			# Active playser may mark white sum on any color
			resp = input(f"{active_player.name} (active): mark white sum {white_sum}? (R/Y/G/B or Enter to skip): ").strip().upper()
			if resp in COLORS:
				if active_player.sheet.mark(resp, white_sum):
					print(f"{active_player.name} marked white sum {white_sum} on {resp}.")
					number_allowed += 1
				else:
					print("Cannot mark white sum there (either invalid position or violates ordering).")
			else:
				number_allowed += 1  # skip
				check_penalties += 1
				print(check_penalties)
			# Active player may mark a colored die combined with a chosen white die (1 or 2)
		number_allowed = 0
		while number_allowed == 0:	
			resp = input(f"{active_player.name} (active): mark colored+white? Enter color (R/Y/G/B) or Enter to skip: ").strip().upper()
			number_allowed += 1  # skip
			check_penalties += 1
			print(check_penalties)
			if resp in COLORS:
				color = resp
				w_choice = input(f"Which white die to use? 1 ({w1}) or 2 ({w2}) — enter 1 or 2: ").strip()
				if w_choice == '1':
					number = w1 + colors[color]
				elif w_choice == '2':
					number = w2 + colors[color]
				else:
					number_allowed += 1  # skip
					check_penalties += 1
					print(check_penalties)
					number = None
				if number is not None:
					if active_player.sheet.mark(color, number):
						print(f"{active_player.name} marked {number} on {color} using white{w_choice}.")
						number_allowed += 1
					else:
						print("Cannot mark that number on that color (already marked or violates ordering).")

		# Non-active players may mark the white sum on their own sheets
		for i, p in enumerate(self.players):
			if i == self.active:
				continue
			number_allowed = 0
			while number_allowed == 0:
				# Each non-active player may mark the white sum on any color
				resp = input(f"{p.name}: mark white sum {white_sum}? (R/Y/G/B or Enter to skip): ").strip().upper()
				if resp in COLORS:
					if p.sheet.mark(resp, white_sum):
						print(f"{p.name} marked white sum {white_sum} on {resp}.")
						number_allowed += 1
					else:
						print('Cannot mark white sum there (already marked or violates ordering).')
				else:
					number_allowed += 1  # skip
		# Show each player's sheet and scores neatly
		print('\n' + '=' * 60)
		for p in self.players:
			print(f"Player: {p.name}   Penalties: {p.penalties}   Score: {p.total_score()}")
			print(p.sheet)
			print('-' * 60)

		# Ask active player if they want to take a penalty voluntarily (or when they couldn't mark)
		print(check_penalties)
	#	pen = input(f"{active_player.name}: take a penalty? (y/N): ").strip().lower()
	#	if pen == 'y':
		if check_penalties == 2:
			active_player.add_penalty()
			print(f"{active_player.name} received a penalty. Total penalties: {active_player.penalties}")
		else:
			print(f"{active_player.name} did not take a penalty.")
			check_penalties = 0
		# Advance to next player
		self.active = (self.active + 1) % len(self.players)

	def run(self):
		print('Starting Qwixx (simplified CLI) — per-player sheets enforced')
		print('Enter q at any prompt to quit early.')
		while True:
			try:
				self.play_turn()
			except (KeyboardInterrupt, EOFError):
				print('\nExiting game.')
				break
			cont = input('Continue? (Y to continue, any other to stop): ').strip().lower()
			if cont != 'y':
				break


def main():
	print('Qwixx CLI — per-player sheets')
	while True:
		n = input('How many players (1-6)? ').strip()
		if not n:
			continue
		if n.lower() == 'q':
			return
		try:
			num = int(n)
			if 1 <= num <= 6:
				break
		except ValueError:
			pass
		print('Please enter a number between 1 and 6.')

	names = []
	for i in range(num):
		nm = input(f'Name for player {i+1} (default Player{i+1}): ').strip()
		if not nm:
			nm = f'Player{i+1}'
		names.append(nm)

	game = QwixxGame(names)
	game.run()


if __name__ == '__main__':
	main()
