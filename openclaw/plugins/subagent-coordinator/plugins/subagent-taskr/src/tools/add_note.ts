/**
 * add_note Tool
 *
 * Adds a note to a task for context tracking, findings, progress, etc.
 * Notes are permanent records attached to tasks.
 */

import type { TaskStoreService, Note } from "../services/task_store";

export type NoteType = "context" | "finding" | "progress" | "file_list" | "error";

export interface AddNoteInput {
  taskId: string;
  content: string;
  type: NoteType;
  author?: string;
}

export interface AddNoteOutput {
  noteId: string;
  taskId: string;
  createdAt: number;
}

export function createAddNoteTool(
  taskStore: TaskStoreService
) {
  return async (input: AddNoteInput): Promise<AddNoteOutput> => {
    const { taskId, content, type, author } = input;

    // Validate task exists
    const task = taskStore.get(taskId);
    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    // Validate content
    if (!content || content.trim().length === 0) {
      throw new Error("Note content cannot be empty");
    }

    // Validate type
    const validTypes: NoteType[] = ["context", "finding", "progress", "file_list", "error"];
    if (!validTypes.includes(type)) {
      throw new Error(`Invalid note type: ${type}. Valid types: ${validTypes.join(", ")}`);
    }

    const noteId = taskStore.addNote(taskId, {
      content: content.trim(),
      type,
      author
    });

    return {
      noteId,
      taskId,
      createdAt: Date.now()
    };
  };
}

/**
 * Format a note for display
 */
export function formatNote(note: Note): string {
  const lines = [
    `**[${note.type.toUpperCase()}]** ${new Date(note.createdAt).toLocaleString()}`
  ];

  if (note.author) {
    lines.push(`_By: ${note.author}_`);
  }

  lines.push("");
  lines.push(note.content);

  if (note.updatedAt) {
    lines.push("");
    lines.push(`_Updated: ${new Date(note.updatedAt).toLocaleString()}_`);
  }

  return lines.join("\n");
}

/**
 * Filter notes by type
 */
export function filterNotesByType(notes: Note[], type: NoteType): Note[] {
  return notes.filter(n => n.type === type);
}

/**
 * Search notes by content
 */
export function searchNotes(notes: Note[], query: string, caseSensitive = false): Note[] {
  const searchQuery = caseSensitive ? query : query.toLowerCase();

  return notes.filter(note => {
    const content = caseSensitive ? note.content : note.content.toLowerCase();
    return content.includes(searchQuery);
  });
}
