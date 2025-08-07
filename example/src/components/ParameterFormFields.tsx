import React from 'react'
import {
  View,
  Text,
  TextInput,
  Switch,
  TouchableOpacity,
  StyleSheet,
} from 'react-native'
import { CommonStyles } from '../styles/commonStyles'

const styles = StyleSheet.create({
  paramGroup: CommonStyles.paramGroup,
  paramLabel: CommonStyles.paramLabel,
  paramDescription: CommonStyles.paramDescription,
  textInput: CommonStyles.textInput,
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  switchInfo: {
    flex: 1,
    marginRight: 12,
  },
  stopSequenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  stopSequenceInput: {
    flex: 1,
    marginRight: 8,
  },
  removeButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  removeButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  addButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 10,
    borderRadius: 8,
    marginTop: 8,
  },
  addButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
})

interface ParameterTextInputProps {
  label: string
  description: string
  value?: string
  onChangeText: (text: string) => void
  placeholder?: string
  keyboardType?: 'default' | 'numeric' | 'decimal-pad'
}

export function ParameterTextInput({
  label,
  description,
  value,
  onChangeText,
  placeholder,
  keyboardType = 'default',
}: ParameterTextInputProps) {
  return (
    <View style={styles.paramGroup}>
      <Text style={styles.paramLabel}>{label}</Text>
      <Text style={styles.paramDescription}>{description}</Text>
      <TextInput
        style={styles.textInput}
        value={value}
        onChangeText={onChangeText}
        keyboardType={keyboardType}
        placeholder={placeholder}
      />
    </View>
  )
}

interface ParameterSwitchProps {
  label: string
  description: string
  value: boolean
  onValueChange: (value: boolean) => void
}

export function ParameterSwitch({
  label,
  description,
  value,
  onValueChange,
}: ParameterSwitchProps) {
  return (
    <View style={styles.paramGroup}>
      <View style={styles.switchRow}>
        <View style={styles.switchInfo}>
          <Text style={styles.paramLabel}>{label}</Text>
          <Text style={styles.paramDescription}>{description}</Text>
        </View>
        <Switch
          value={value}
          onValueChange={onValueChange}
          trackColor={{ false: '#E0E0E0', true: '#007AFF' }}
          thumbColor={value ? '#FFFFFF' : '#FFFFFF'}
        />
      </View>
    </View>
  )
}

interface StopSequenceFieldProps {
  stopSequences: string[]
  onUpdateStopSequence: (index: number, value: string) => void
  onRemoveStopSequence: (index: number) => void
  onAddStopSequence: () => void
}

export function StopSequenceField({
  stopSequences,
  onUpdateStopSequence,
  onRemoveStopSequence,
  onAddStopSequence,
}: StopSequenceFieldProps) {
  return (
    <View style={styles.paramGroup}>
      <Text style={styles.paramLabel}>Stop Sequences</Text>
      <Text style={styles.paramDescription}>
        Text sequences that will stop generation when encountered. Common
        examples: `User:`, `Llama:`, `Assistant:`
      </Text>

      {stopSequences.map((stopSeq, index) => (
        <View key={index} style={styles.stopSequenceContainer}>
          <TextInput
            style={[styles.textInput, styles.stopSequenceInput]}
            value={stopSeq}
            onChangeText={(text) => onUpdateStopSequence(index, text)}
            placeholder="Enter stop sequence"
          />
          <TouchableOpacity
            style={styles.removeButton}
            onPress={() => onRemoveStopSequence(index)}
          >
            <Text style={styles.removeButtonText}>Remove</Text>
          </TouchableOpacity>
        </View>
      ))}

      <TouchableOpacity style={styles.addButton} onPress={onAddStopSequence}>
        <Text style={styles.addButtonText}>Add Stop Sequence</Text>
      </TouchableOpacity>
    </View>
  )
}
