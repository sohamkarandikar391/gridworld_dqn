import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_gridworld_image(filename='gridworld.pdf'):
    """
    Creates a clean image of the 3x4 gridworld with state labels.
    
    Grid layout:
    - Wall at position (1, 1)
    - Terminal states: (0, 3) with +1 reward, (1, 3) with -1 reward
    - All other cells are regular states
    """
    # Grid parameters
    rows = 3
    cols = 4
    walls = [(1, 1)]
    terminals = {(0, 3): 1, (1, 3): -1}  # (row, col): reward
    
    # Create state numbering (same as GridWorld class logic)
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Mark walls
    for wall in walls:
        row, col = wall
        grid[row][col] = -1
    
    # Create state mapping (bottom-left to top-right, row-first)
    pos_to_state = {}
    state_num = 0
    
    # Start from bottom row (row = rows-1) and go up
    for row in range(rows - 1, -1, -1):
        for col in range(cols):
            if grid[row][col] != -1:  # Not a wall
                pos_to_state[(row, col)] = state_num
                state_num += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    
    # Draw grid cells
    for row in range(rows):
        for col in range(cols):
            if (row, col) in walls:
                # Wall (black)
                ax.add_patch(patches.Rectangle(
                    (col-0.5, row-0.5), 1, 1, 
                    facecolor='black', edgecolor='darkgray', linewidth=2
                ))
            elif (row, col) in terminals:
                # Terminal states
                if terminals[(row, col)] > 0:
                    # Positive reward (green)
                    ax.add_patch(patches.Rectangle(
                        (col-0.5, row-0.5), 1, 1, 
                        facecolor='lightgreen', edgecolor='darkgray', linewidth=2
                    ))
                    # Add reward label
                    ax.text(col, row, '+1', fontsize=24, ha='center', va='center', 
                           fontweight='bold', color='darkgreen')
                else:
                    # Negative reward (red)
                    ax.add_patch(patches.Rectangle(
                        (col-0.5, row-0.5), 1, 1, 
                        facecolor='lightcoral', edgecolor='darkgray', linewidth=2
                    ))
                    # Add reward label
                    ax.text(col, row, '-1', fontsize=24, ha='center', va='center', 
                           fontweight='bold', color='darkred')
                
                # Add state number in bottom right corner
                state_num = pos_to_state[(row, col)]
                ax.text(col + 0.35, row - 0.35, f'{state_num}', 
                       fontsize=14, ha='right', va='bottom', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.8))
            else:
                # Regular state (white)
                ax.add_patch(patches.Rectangle(
                    (col-0.5, row-0.5), 1, 1, 
                    facecolor='white', edgecolor='darkgray', linewidth=2
                ))
                
                # Add state number in bottom right corner
                state_num = pos_to_state[(row, col)]
                ax.text(col + 0.35, row - 0.35, f'{state_num}', 
                       fontsize=14, ha='right', va='bottom', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', 
                                edgecolor='gray', alpha=0.8))
    
    # Set title
    ax.set_title('3×4 GridWorld Environment', fontsize=20, fontweight='bold', pad=20)
    
    # Configure axes
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f'{i}' for i in range(cols)], fontsize=12)
    ax.set_yticklabels([f'{i}' for i in range(rows-1, -1, -1)], fontsize=12)  # 2, 1, 0 from top to bottom
    ax.set_xlabel('x-coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('y-coordinate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add START label at position (row=2 in display, which is row 0 logically) at col=0
    ax.text(0, 2, 'START', fontsize=16, ha='center', va='center', 
       fontweight='bold', color='black', 
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                edgecolor='black', alpha=0.7, linewidth=2))
    
    # Invert y-axis to match typical grid representation (0,0 at top-left)
    ax.invert_yaxis()
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='darkgray', label='Regular State'),
        patches.Patch(facecolor='lightgreen', edgecolor='darkgray', label='Goal (+1)'),
        patches.Patch(facecolor='lightcoral', edgecolor='darkgray', label='Penalty (-1)'),
        patches.Patch(facecolor='black', edgecolor='darkgray', label='Wall')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
             fontsize=12, framealpha=0.9, ncol=4)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"GridWorld image saved as '{filename}'")
    plt.show()

if __name__ == "__main__":
    create_gridworld_image('gridworld.pdf')