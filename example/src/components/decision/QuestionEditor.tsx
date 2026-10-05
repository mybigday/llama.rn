import React from 'react'
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native'
import Icon from '@react-native-vector-icons/material-design-icons'
import { useTheme } from '../../contexts/ThemeContext'
import type {
  DecisionQuestionType,
  EditableQuestion,
} from '../../features/decisionHelpers'

const TYPE_LABELS: Array<{ type: DecisionQuestionType; label: string }> = [
  { type: 'choice', label: 'Choice' },
  { type: 'score', label: 'Score' },
  { type: 'noul', label: 'Yes / No' },
]

interface QuestionEditorProps {
  index: number
  question: EditableQuestion
  onChange: (question: EditableQuestion) => void
  onRemove: () => void
}

export function QuestionEditor({
  index,
  question,
  onChange,
  onRemove,
}: QuestionEditorProps) {
  const { theme } = useTheme()
  const styles = createStyles(theme.colors)
  const update = (fields: Partial<EditableQuestion>) =>
    onChange({ ...question, ...fields })

  const renderRemoveButton = (onPress: () => void, disabled: boolean) => (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      style={styles.rowButton}
      hitSlop={8}
    >
      <Icon
        name="close"
        size={18}
        color={disabled ? theme.colors.disabled : theme.colors.textSecondary}
      />
    </TouchableOpacity>
  )

  const renderAddButton = (label: string, onPress: () => void) => (
    <TouchableOpacity style={styles.addButton} onPress={onPress}>
      <Icon name="plus" size={16} color={theme.colors.primary} />
      <Text style={styles.addButtonText}>{label}</Text>
    </TouchableOpacity>
  )

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={styles.indexBadge}>
          <Text style={styles.indexText}>{index + 1}</Text>
        </View>
        <TextInput
          style={styles.idInput}
          value={question.id}
          onChangeText={(id) => update({ id })}
          placeholder="question_id"
          placeholderTextColor={theme.colors.textSecondary}
          autoCapitalize="none"
          autoCorrect={false}
        />
        <TouchableOpacity onPress={onRemove} hitSlop={8}>
          <Icon name="delete-outline" size={22} color={theme.colors.error} />
        </TouchableOpacity>
      </View>

      <View style={styles.segmented}>
        {TYPE_LABELS.map(({ type, label }) => {
          const selected = question.type === type
          return (
            <TouchableOpacity
              key={type}
              style={[styles.segment, selected && styles.segmentSelected]}
              onPress={() => update({ type })}
            >
              <Text
                style={[
                  styles.segmentText,
                  selected && styles.segmentTextSelected,
                ]}
              >
                {label}
              </Text>
            </TouchableOpacity>
          )
        })}
      </View>

      <TextInput
        style={styles.instructionsInput}
        value={question.instructions}
        onChangeText={(instructions) => update({ instructions })}
        placeholder={
          question.type === 'noul'
            ? 'A statement to judge, e.g. "The customer is angry"'
            : 'The question, e.g. "Which team should handle this?"'
        }
        placeholderTextColor={theme.colors.textSecondary}
        multiline
      />

      {question.type === 'choice' && (
        <View>
          <Text style={styles.sectionLabel}>Options</Text>
          {question.options.map((option, i) => (
            <View key={i} style={styles.row}>
              <TextInput
                style={[styles.input, styles.keyInput]}
                value={option.key}
                onChangeText={(key) =>
                  update({
                    options: question.options.map((o, j) =>
                      j === i ? { ...o, key } : o,
                    ),
                  })
                }
                placeholder="key"
                placeholderTextColor={theme.colors.textSecondary}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <TextInput
                style={[styles.input, styles.flexInput]}
                value={option.description}
                onChangeText={(description) =>
                  update({
                    options: question.options.map((o, j) =>
                      j === i ? { ...o, description } : o,
                    ),
                  })
                }
                placeholder="description (optional)"
                placeholderTextColor={theme.colors.textSecondary}
              />
              {renderRemoveButton(
                () =>
                  update({
                    options: question.options.filter((_, j) => j !== i),
                  }),
                question.options.length <= 1,
              )}
            </View>
          ))}
          {renderAddButton('Add option', () =>
            update({
              options: [...question.options, { key: '', description: '' }],
            }),
          )}
        </View>
      )}

      {question.type === 'score' && (
        <View>
          <Text style={styles.sectionLabel}>Levels, lowest first</Text>
          {question.levels.map((level, i) => (
            <View key={i} style={styles.row}>
              <Text style={styles.levelIndex}>{i}</Text>
              <TextInput
                style={[styles.input, styles.flexInput]}
                value={level}
                onChangeText={(text) =>
                  update({
                    levels: question.levels.map((l, j) => (j === i ? text : l)),
                  })
                }
                placeholder={`level ${i}`}
                placeholderTextColor={theme.colors.textSecondary}
              />
              {renderRemoveButton(
                () =>
                  update({ levels: question.levels.filter((_, j) => j !== i) }),
                question.levels.length <= 2,
              )}
            </View>
          ))}
          {question.levels.length < 10 &&
            renderAddButton('Add level', () =>
              update({ levels: [...question.levels, ''] }),
            )}
        </View>
      )}

      {question.type === 'noul' && (
        <View>
          <Text style={styles.sectionLabel}>What yes / no mean (optional)</Text>
          <View style={styles.row}>
            <Text style={styles.noulLabel}>Yes</Text>
            <TextInput
              style={[styles.input, styles.flexInput]}
              value={question.noulTrue}
              onChangeText={(noulTrue) => update({ noulTrue })}
              placeholder="e.g. a drink is mentioned"
              placeholderTextColor={theme.colors.textSecondary}
            />
          </View>
          <View style={styles.row}>
            <Text style={styles.noulLabel}>No</Text>
            <TextInput
              style={[styles.input, styles.flexInput]}
              value={question.noulFalse}
              onChangeText={(noulFalse) => update({ noulFalse })}
              placeholder="e.g. no drink"
              placeholderTextColor={theme.colors.textSecondary}
            />
          </View>
        </View>
      )}
    </View>
  )
}

function createStyles(colors: ReturnType<typeof useTheme>['theme']['colors']) {
  const input = {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.inputBackground,
  }
  return StyleSheet.create({
    card: {
      backgroundColor: colors.surface,
      borderRadius: 12,
      padding: 12,
      marginBottom: 12,
      borderWidth: 1,
      borderColor: colors.border,
    },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 10,
      marginBottom: 10,
    },
    indexBadge: {
      width: 24,
      height: 24,
      borderRadius: 12,
      backgroundColor: colors.primary,
      alignItems: 'center',
      justifyContent: 'center',
    },
    indexText: { color: colors.white, fontSize: 12, fontWeight: '700' },
    idInput: {
      ...input,
      flex: 1,
      fontFamily: 'monospace',
      paddingVertical: 6,
    },
    segmented: {
      flexDirection: 'row',
      borderRadius: 8,
      borderWidth: 1,
      borderColor: colors.border,
      overflow: 'hidden',
      marginBottom: 10,
    },
    segment: { flex: 1, paddingVertical: 7, alignItems: 'center' },
    segmentSelected: { backgroundColor: colors.primary },
    segmentText: { fontSize: 13, fontWeight: '600', color: colors.text },
    segmentTextSelected: { color: colors.white },
    instructionsInput: { ...input, minHeight: 44, marginBottom: 6 },
    sectionLabel: {
      fontSize: 12,
      fontWeight: '600',
      color: colors.textSecondary,
      marginTop: 6,
      marginBottom: 6,
    },
    row: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      marginBottom: 6,
    },
    input,
    keyInput: { width: 96, fontFamily: 'monospace' },
    flexInput: { flex: 1 },
    rowButton: { padding: 4 },
    levelIndex: {
      width: 18,
      textAlign: 'center',
      fontSize: 13,
      fontWeight: '600',
      color: colors.textSecondary,
    },
    noulLabel: {
      width: 32,
      fontSize: 13,
      fontWeight: '600',
      color: colors.textSecondary,
    },
    addButton: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 4,
      paddingVertical: 6,
    },
    addButtonText: { color: colors.primary, fontSize: 13, fontWeight: '600' },
  })
}
